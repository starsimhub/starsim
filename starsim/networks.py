"""
Networks and mixing pools: who meets whom, and so who can infect whom

A network holds the contacts (edges) between people, over which a disease spreads. Mixing pools (`ss.MixingPool`, `ss.networks.MixingPools`) and contact matrices (`ss.contacts()`) spread it between groups instead, without edges. These exist so that a model's contacts are separate from its diseases, and one network can carry several diseases. In a compartmental run, a network is replaced by what it implies: contact rates, a matrix, or a model of partner change.

Used with: agents and compartments; declared and coded modules
"""
import re
import itertools
import functools
import numpy as np
import numba as nb
import scipy.sparse as spsp
import sciris as sc
import starsim as ss
plt = sc.lazyimport('matplotlib.pyplot') # Imported when first used, since most runs never plot
import pandas as pd

# This has a significant impact on runtime, surprisingly
ss_float = ss.dtypes.float
ss_int = ss.dtypes.int
_ = None


@nb.jit(cache=True)
def fisher_yates_shuffle(arr, randvals):
    """ In-place Fisher-Yates shuffle using precomputed uniform random values.

    Equivalent to `np.random.permutation` (a uniform random shuffle), but ~1.3x faster
    for the large arrays used in network generation. `randvals` should be uniform on
    [0, 1) and the same length as `arr`.

    Note: `randvals` is consumed in order (`randvals[k]` for the k-th swap), not indexed
    by position. This means that changing the array length (e.g. adding one agent) shifts
    every swap, so the shuffle is *not* random-number safe -- matching `ss.RandomNet(method='exact')`'s
    intended behavior (see `ss.RandomNet(method='safe')` for the CRN-safe version).
    """
    n = arr.shape[0]
    for k in range(n - 1):
        i = n - 1 - k # Position to swap, working from the top down
        j = int(randvals[k] * (i + 1))
        tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp
    return arr


@nb.jit(cache=True) # No fastmath: a stray NaN must compare False (no transmission), not be assumed absent
def nb_transmit(src, trg, rel_trans, rel_sus, hazard, randvals):
    """ Optimized transmission kernel: returns the (source, target) UIDs of transmitting edges.

    Edge i transmits with probability 1 − exp(−rel_trans[src]·rel_sus[trg]·hazard[i]): every factor scales the hazard in the step,
    so one contact infects at most once, whichever factor a model puts its effect in (see `Network.transmit()`).
    Fuses the gather/multiply/compare *and* the UID extraction into a single branchless pass.
    This avoids a full-length boolean temporary and two separate gather passes
    (`trg[mask]`, `src[mask]`) -- ~1.4x faster at typical transmission rates, more when
    transmission is common. UIDs are emitted in edge order, preserving CRN behavior.
    """
    n = src.shape[0]
    src_out = np.empty(n, dtype=np.int64) # int64 = uid dtype, so the caller can .view(uids) without a copy
    trg_out = np.empty(n, dtype=np.int64)
    m = 0
    for i in range(n):
        x = float(rel_trans[src[i]] * rel_sus[trg[i]] * hazard[i])
        transmitted = x > randvals[i] and -np.expm1(-x) > randvals[i] # 1 − exp(−x) ≤ x, so most edges are ruled out without the exp
        src_out[m] = src[i] # Written every iteration (branchless); kept only if m advances
        trg_out[m] = trg[i]
        m += transmitted
    return src_out[:m], trg_out[:m]


# %% Helpers

@nb.jit(cache=True)
def _mark_partners(p1, p2, is_ind, is_partner):
    """ Mark the partners of the agents flagged in is_ind, in both directions of each edge; used by find_contacts() """
    for i in range(len(p1)):
        if is_ind[p1[i]]:
            is_partner[p2[i]] = True
        if is_ind[p2[i]]:
            is_partner[p1[i]] = True
    return


def find_contacts(p1, p2, inds, as_array=False):
    """
    Find the partners of the specified agents in an edge list; see also `Network.find_contacts()`.

    By default, a set is returned so that custom tracing interventions can efficiently add extra
    people; with `as_array=True`, a sorted array of unique UIDs is returned instead, which is faster.

    Args:
        p1 (array): the first agent of each edge
        p2 (array): the second agent of each edge
        inds (array): the agents whose partners to find
        as_array (bool): if True, return a sorted array rather than a set
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    inds = np.asarray(inds, dtype=np.int64)

    # Make lookup tables of which people are in inds, and which are their partners (much faster than np.isin() or a set)
    n = max(np.max(p1, initial=-1), np.max(p2, initial=-1), np.max(inds, initial=-1)) + 1
    is_ind = np.zeros(n, dtype=bool)
    is_ind[inds] = True
    is_partner = np.zeros(n, dtype=bool)
    _mark_partners(p1, p2, is_ind, is_partner)

    partners = np.flatnonzero(is_partner).astype(ss.dtypes.int, copy=False)
    if not as_array:
        partners = set(partners.tolist())
    return partners


def standardize_netkey(key):
    """
    The name of a network, standardized, so that a beta by route matches it however it is written

    A network's name can be upper or lowercase, end in 'net' or not, and have a number
    added to make it unique: 'RandomNet' and 'randomnet' both become 'random', and
    'randomnet_1' becomes 'random_1'.
    """
    return re.sub(r'net(_\d+)?$', r'\1', key.lower())


def route_beta(betas, route):
    """
    The value of a beta for each route (a dict by route name) on one route, or None if it has none

    The dict may give `default`, the value of every route it does not name, e.g.
    `dict(sexualnet=ss.peract(0.01), msm=ss.peract(0.1), default=0)`.
    """
    keys = {standardize_netkey(k):v for k,v in betas.items() if k != 'default'}
    return keys.get(standardize_netkey(route), betas.get('default'))


# %% General network classes

def cond_str(cond):
    """ A condition over people, for messages: e.g. 'people.female' """
    if isinstance(cond, ss.conditions.Cond) and cond.op == 'people':
        return f'people.{cond.args[0]}'
    elif isinstance(cond, ss.conditions.Cond) and cond.op == '~' and isinstance(cond.args[0], ss.conditions.Cond) and cond.args[0].op == 'people':
        return f'not people.{cond.args[0].args[0]}'
    return getattr(cond, '__name__', str(cond))


def check_edges(edges):
    """ The weight of each edge is `weight`; a column `beta` (Starsim 3's name for it) is an error, since a disease's beta is the disease's """
    if edges is not None and 'beta' in edges:
        errormsg = "the weight of each edge of a network is weight= (e.g. weight=np.ones(n)), not beta=; a disease's beta on a network is the disease's, by route (e.g. ss.SIS(beta=dict(mynet=0.05)))"
        raise TypeError(errormsg)
    return


def people_mask(sim, cond):
    """
    Who a condition over people holds for, as a boolean per agent (indexed by UID)

    Args:
        cond (Cond/str/func): a condition (`ss.person.pwid`), the name of a level of a stratification or of a boolean property of people (`'fsw'`), or a function of the sim
    """
    if isinstance(cond, str):
        strat = next((strat for strat in sim.groups.strats if cond in strat.labels), None)
        cond = ss.conditions.Cond('dim', strat.name) == cond if strat is not None else ss.conditions.Cond('people', cond)
    return np.asarray(ss.conditions.Selector(sim).select(cond), dtype=bool)


class Route(ss.Module):
    """
    A way a disease spreads between people: the base class of networks and mixing pools

    A disease's beta can be given for each route, by its name (e.g. `ss.SIS(beta=dict(randomnet=0.05))`).
    """
    sets_unit = False # Its typed defaults (e.g. how long partnerships last) do not set the unit of the model's bare numbers: see ss.solvers.choose.init_unit()
    one_step = False # Whether every edge lasts one step: the edges of the dead then end at the next step anyway, so People.remove_dead() leaves them (see RandomNet)
    def agent_reasons(self, sim):
        """ A route works on groups, unless it says otherwise """
        return []

    def compute_transmission(self, rel_sus, rel_trans, disease_beta, disease=None):
        """ The people infected over this route in a step, for a disease written as code; a route other than a network (e.g. a mixing pool) defines it

        Args:
            rel_sus (array): the relative susceptibility of each agent
            rel_trans (array): the relative transmissibility of each agent
            disease_beta (float): the disease's transmission probability
            disease (Disease): the disease
        """
        errormsg = 'compute_transmission() must be defined by Route subclasses'
        raise NotImplementedError(errormsg)


class Network(Route):
    """
    A network of contacts (edges) between people, over which diseases spread, and the methods
    for changing them. This base class has fixed edges, given by hand; the built-in networks,
    which make their own and change them over time, are its subclasses, e.g. `ss.RandomNet`,
    `ss.DegreeNet`, `ss.SexualNet`, and `ss.PrenatalNet`. See `ss.Disease` for how
    transmission uses the edges.

    Each edge has the two people it joins (`p1` and `p2`) and a weight; subclasses add other
    columns, e.g. how long it lasts (`dur`) or the acts in it (`acts`).

    Note: CRN safe, with the edges given.

    Args:
        pars (dict): parameters, for subclasses that define them
        edges (dict/dataframe): the edges of the network, as an alternative to passing them as keyword arguments
        p1 (array): an array of length N, the number of connections in the network, with the indices of people on one side of the connection.
        p2 (array): an array of length N, the number of connections in the network, with the indices of people on the other side of the connection.
        weight (array): the weight of each edge, which scales the hazard of transmission over it (1 if not given; see `net_beta()`); a disease's own beta on the network is on the disease
        one_way (bool): whether it transmits only from p1 to p2, as mother-to-child networks do (by default, both ways)
        kwargs (dict): standard module arguments (e.g. `name`, `label`, `dt`) are passed to `update_pars()`; other keys are copied directly into the network as columns of the edges

    All the columns of the edges must be arrays of the same length when the sim is
    initialized, though not all need be given when the network is made.

    Examples:
        ```python
        # Generate an average of 10 contacts for 1000 people
        n_contacts_pp = 10
        n_people = 1000
        n = n_contacts_pp * n_people
        p1 = np.random.randint(n_people, size=n)
        p2 = np.random.randint(n_people, size=n)
        weight = np.ones(n)
        network = ss.Network(p1=p1, p2=p2, weight=weight, label='rand')
        network = ss.Network(edges=dict(p1=p1, p2=p2, weight=weight), label='rand') # Alternate method

        # Convert one network to another with extra columns
        index = np.arange(n)
        self_conn = p1 == p2
        network2 = ss.Network(edges=network.edges, index=index, self_conn=self_conn, label=network.label)
        ```
    """

    def __init__(self, pars=None, edges=None, **kwargs):
        super().__init__()
        self.meta = sc.obj( # The columns of the edges, and their types
            p1 = ss_int,
            p2 = ss_int,
            weight = ss_float,
        )

        self.edge_defaults = dict(weight=1.0) # Values of edge columns that need not be given when edges are added (e.g. condoms=0 on a sexual network)
        self.edges = sc.obj()
        self.participant = ss.BoolArr('participant')
        self.agent_edges = None # The edges of each agent, once the edges have stayed the same for a step (see edge_index())

        # Standard module arguments are parameters; the rest are edges
        if 'sides' in kwargs: # The kinds of people at the two ends of its edges; see directions
            self.sides = kwargs.pop('sides')
        if 'one_way' in kwargs: # Whether it transmits only from p1 to p2
            self.one_way = kwargs.pop('one_way')
        standard_args = ss.modules.module_args + ss.Timeline.time_args
        edges = dict(edges) if edges is not None else {} # dict() also handles dataframes
        edges.update({k:kwargs.pop(k) for k in list(kwargs) if k not in standard_args})
        check_edges(edges)
        for key, value in edges.items():
            self.edges[key] = np.array(value, dtype=self.meta.get(key)) # Overwrite dtype if supplied, else keep original
        self.update_pars(pars, **kwargs)
        return

    @property
    def p1(self):
        """ The first half of a network edge (person 1) """
        return self.edges['p1'] if 'p1' in self.edges else None

    @property
    def p2(self):
        """ The second half of a network edge (person 2) """
        return self.edges['p2'] if 'p2' in self.edges else None

    @property
    def weight(self):
        """ The weight of each edge, which scales the hazard of transmission over it (see `net_beta()`) """
        return self.edges['weight'] if 'weight' in self.edges else None

    #%% Sides, directions, and acts

    sides = None # The kinds of people at the two ends of its edges: a dict of a short label to a condition over people (e.g. dict(m=~ss.person.female, f=ss.person.female)), from which its directions follow
    same_side = False # Whether an edge can join two people on the same side (e.g. partnerships between anyone, by sex)
    one_way = False # Whether it transmits only from p1 to p2 (mother to child), whose sides are then where they are in an edge

    @property
    def has_acts(self):
        """ Whether the network has acts in each partnership (`edges.acts`), so that a disease's beta on it is a probability per act (see `ss.peract`) """
        return 'acts' in self.meta

    @property
    def directions(self):
        """
        The directions it can transmit in, `<side of the infectious partner>2<side of the susceptible one>`,
        e.g. `['m2f', 'f2m']` on `ss.SexualNet`; these are the keys of an `ss.peract` on it. A network with no
        sides has none, and takes one value.
        """
        sides = list(self.sides or {})
        if self.one_way:
            return [f'{sides[0]}2{sides[1]}'] if len(sides) == 2 else []
        return [f'{a}2{b}' for a in sides for b in sides if a != b] + ([f'{a}2{a}' for a in sides] if self.same_side else [])

    def sides_str(self):
        """ The sides, for messages and describe(), e.g. 'sides m and f (m: male, f: female)' """
        if not self.sides:
            return 'no sides'
        what = ', '.join(f'{k}: {cond_str(v)}' for k,v in self.sides.items())
        return f'sides {sc.strjoin(list(self.sides), sep=" and ")} ({what})'

    def side_raw(self):
        """ The side of each agent, as a position in `sides` (-1 for neither), indexed by UID """
        sim = self.sim
        out = np.full(sim.people.uid.len_tot, -1, dtype=np.int8)
        for i,cond in enumerate(self.sides.values()):
            mask = people_mask(sim, cond)
            if np.any(mask & (out >= 0)):
                errormsg = f'Some people are on two sides of "{self.name}" ({sc.strjoin(list(self.sides))}); each person can be on one'
                raise ValueError(errormsg)
            out[mask] = i
        return out

    def edge_directions(self, src, trg):
        """ The direction of each edge from `src` to `trg`, as a position in `directions` (0 for a network with no sides); raises a `ValueError` for an edge with no direction """
        if not self.sides or self.one_way:
            return np.zeros(len(src), dtype=np.int8) # One direction, or edges from p1 to p2
        dirs = self.directions
        names = list(self.sides)
        n = len(names)
        table = np.full((n + 1, n + 1), -1, dtype=np.int8) # The last row and column are for people on neither side
        for a,b in itertools.product(range(n), range(n)):
            d = f'{names[a]}2{names[b]}'
            if d in dirs:
                table[a, b] = dirs.index(d)
        side = self.side_raw()
        out = table[side[src], side[trg]]
        if np.any(out < 0):
            k = np.flatnonzero(out < 0)[0]
            hint = "; for partnerships between men, give between='mm' (in a subclass, e.g. super().__init__(between='mm'))" if isinstance(self, ss.SexualNet) else ''
            raise ValueError(f'"{self.name}" has a partnership from {src[k]} to {trg[k]}, whose sides give no direction it has ({sc.strjoin(dirs)}; {self.name} has {self.sides_str()}){hint}')
        return out

    def transmit(self, rel_trans, rel_sus, edge_hazard, rng, prob=None):
        """
        The edges that transmit in a step, in each direction the network transmits in (p1 to p2, then p2 to p1 unless it is one way)

        An edge from `source` to `target` transmits with probability 1 − exp(−`rel_trans[source]·rel_sus[target]·hazard`), or `prob()` if it is given:
        each factor scales the hazard over the step, so a contact infects at most once in it, wherever a model puts an effect.
        Random numbers are drawn only for the edges that can transmit: with CRN, each edge's number depends only on its source
        and target, so this gives the same result as drawing one for every edge. The beta is worked out only for the edges from
        someone infectious to someone susceptible, which are found for both directions in one pass over the edges, or, once the
        edges have stayed the same for a step (`edge_index()`), from the edges of the infectious (or the susceptible) alone, if
        they have few; either way they are in the order of the edges.

        Args:
            rel_trans (array): the transmissibility of each agent, by UID (0 if not infectious)
            rel_sus (array): the susceptibility of each agent, by UID (0 if not susceptible)
            edge_hazard (func): of the direction (0 or 1), and the indices, sources, and targets of the edges from someone with `rel_trans` not 0
                to someone with `rel_sus` not 0, the hazard of each of them in the step (an array, or a number for all): with a function
                as `prob`, only where it is 0 matters (no transmission); None to skip the direction (no random numbers are drawn)
            rng (`ss.multi_random`): the random numbers, by source and target
            prob (func): of the direction, and the indices, sources, and targets of the edges that can transmit, the probability of each, in place of the one above

        Returns:
            A list of the (source UIDs, target UIDs) that transmit, for each direction not skipped
        """
        out = []
        if not len(self):
            return out
        edges = self.edges
        found = ss.solvers.backends.passes.candidates(edges.p1, edges.p2, rel_trans, rel_sus, two_way=not self.one_way, index=self.edge_index()) # Mother-to-child networks transmit one way
        for d,inds in enumerate(found):
            source, target = [ends[inds] for ends in self.ends(d)]
            hazard = edge_hazard(d, inds, source, target)
            if hazard is None:
                continue
            hazard = np.asarray(hazard, dtype=ss_float) if np.ndim(hazard) == 0 else hazard # One value for all edges
            can = ss.solvers.backends.passes.can_transmit(source, target, rel_trans, rel_sus, hazard) # Of those, the ones with a chance above 0, exactly as over every edge
            if len(can) < len(inds):
                inds, source, target = inds[can], source[can], target[can]
                hazard = hazard[can] if np.ndim(hazard) else hazard
            rvs = rng.rvs(source, target) # Called even if empty, to keep CRN draws in sync
            if prob is None:
                hazard = np.broadcast_to(hazard, len(inds)) if np.ndim(hazard) == 0 else hazard
                source, target = nb_transmit(np.asarray(source), np.asarray(target), rel_trans, rel_sus, hazard, rvs)
            else:
                hit = rvs < prob(d, inds, source, target)
                source, target = np.asarray(source)[hit], np.asarray(target)[hit]
            out.append((source.view(ss.uids), target.view(ss.uids))) # View (no copy): the kernel's output is its own int64 array
        return out

    def edge_index(self, now=False):
        """
        The edges of each agent (see `ss.solvers.backends.passes.edge_index()`), so that transmission from a few people, or contact tracing,
        visits only their edges: None until the edges have stayed the same from one step to the next, so a network whose edges change every
        step never makes one, unless `now` (for agents taken by events, which visit only the edges of the people infectious however often
        the edges change). The edges change when they are assigned again (e.g. `edges.p1 = ...`), as every method that changes them does.
        """
        p1, p2 = self.edges.p1, self.edges.p2
        idx = self.agent_edges
        if idx is not None and idx.lists is not None:
            if idx.p1 is p1 and idx.p2 is p2:
                return idx.lists
            idx = None
        ids = (id(p1), id(p2)) # Until it is made, not the arrays themselves, which would keep a changing network's old edges in memory; an id used again only makes it sooner
        new = idx is None or idx.ids != ids # New edges
        if new:
            self.agent_edges = sc.obj(ids=ids, ti=self.ti, lists=None)
        if len(p1) and (now or (not new and self.ti is not None and idx.ti is not None and self.ti > idx.ti)): # The same as in an earlier step, or wanted now
            n = max(p1.max(), p2.max()) + 1
            self.agent_edges = sc.obj(p1=p1, p2=p2, lists=ss.solvers.backends.passes.edge_index(p1, p2, n))
            return self.agent_edges.lists
        return None

    def init_results(self):
        """ Store network length by default """
        super().init_results()
        self.define_results(
            ss.Result('n_edges', dtype=int, scale=True, label='Number of edges', auto_plot=False)
        )
        return

    def init_pre(self, sim):
        """ Initialize with the sim, initialize the edges, and validate p1 and p2

        Args:
            sim (Sim): the sim being initialized
        """
        super().init_pre(sim)
        n = len(self.edges['p1']) if 'p1' in self.edges else 0
        for key, dtype in self.meta.items(): # The columns not given: their defaults, or empty
            if key not in self.edges:
                self.edges[key] = np.full(n, self.edge_defaults[key], dtype=dtype) if key in self.edge_defaults else np.empty((0,), dtype=dtype)
        self.validate_uids()
        return

    def init_post(self, add_pairs=True):
        """ Add the first edges

        Args:
            add_pairs (bool): whether to add them
        """
        super().init_post()
        if add_pairs: self.add_pairs()
        return

    def agent_reasons(self, sim):
        """ An individual-level network that has an approximation; one with none is refused by the layout of the mixing (see `ss.solvers.Mixing.init()`) """
        return [ss.modules.Preference(f'{type(self).__name__} is an individual-level network')] if self.mean_contacts(sim)[0] is not None else []

    def mean_contacts(self, sim):
        """
        For a compartmental run: the mean number of contacts per person per unit of the sim's time, which homogeneous mixing approximates
        this network by, and where it comes from; or None, and why, if it has no such approximation (the default). A network of
        partnerships has none: its contacts last, so their number per unit time is not its number of partners (it uses the
        partner-change model instead: see `ss.solvers.Mixing.partnerships()`).

        Args:
            sim (Sim): the sim being initialized
        """
        return None, ''

    def __len__(self):
        """ The number of edges """
        try:    return len(self.edges.p1)
        except Exception: return 0 # E.g. before the edges exist, or once shrunk

    def __repr__(self, **kwargs):
        """ The usual brief description, with the number of edges """
        return self.brief(output=True, extra=f'n_edges={len(self)}')

    def __str__(self):
        """ The brief description, and the edges as a dataframe """
        out = self.brief(output=True)
        if not self.initialized or not len(self.edges): # No edges yet, before the sim is initialized
            return out + '\n(no edges yet)'
        out += '\n' + self.to_df().__repr__()
        return out

    def __contains__(self, item):
        """
        Whether a person (by UID) is in any edge of the network

        Args:
            item (int): the person's UID
        """
        return (item in self.edges.p1) or (item in self.edges.p2) # TODO: check if (item in self.members) is faster

    @property
    def members(self):
        """ The UIDs of everyone in at least one edge, sorted """
        return np.unique([self.edges.p1, self.edges.p2]).view(ss.uids)

    def meta_keys(self):
        """ The names of the columns of the edges """
        return self.meta.keys()

    def set_network_states(self, people):
        """
        Set the states of people that the network depends on (nothing by default)

        Some depend on other properties of people, e.g. the age of debut on a sexual network
        varies by sex and over time, so they are set again whenever people are added.

        Args:
            people (People): the people
        """
        pass

    def validate_uids(self):
        """ Make p1 and p2 UID arrays, if they are not """
        for key in ['p1', 'p2']:
            if key in self.edges and not isinstance(self.edges[key], ss.uids):
                self.edges[key] = ss.uids(self.edges[key])
        return

    def validate(self):
        """
        Check the columns of the edges: convert any of the wrong type, and raise an error for any of the wrong length
        """
        if not len(self.edges): # No edges have been added yet
            return
        n = len(self.edges.p1)
        for key, dtype in self.meta.items():
            if dtype and self.edges[key].dtype != dtype:
                self.edges[key] = np.array(self.edges[key], dtype=dtype)
            actual_n = len(self.edges[key])
            if n != actual_n:
                errormsg = f'Expecting length {n} for network key "{key}"; got {actual_n}'
                raise TypeError(errormsg)
        self.validate_uids()
        return

    def eligible(self, who=None):
        """
        The agents who can form edges, as a boolean array: alive, born, and (optionally) those a condition holds for

        Args:
            who (Cond/str/func): if given, only these people: a condition over people (e.g. `(ss.person.age >= 18) & (ss.person.age < 65)`), as for `ss.DegreeNet`
        """
        people = self.sim.people
        eligible = people.alive & (people.age > 0)
        if who is not None:
            eligible = eligible & people_mask(self.sim, who)
        return eligible

    def get_inds(self, inds, remove=False):
        """
        The edges at the given indices, as a dict of arrays by column

        Args:
            inds (int, array, slice): the indices of the edges
            remove (bool): whether to also remove them from the network
        """
        output = {}
        for key in self.meta_keys():
            output[key] = self.edges[key][inds]
            if remove:
                self.edges[key] = np.delete(self.edges[key], inds)
        if remove:
            self.validate_uids()
        return output

    def pop_inds(self, inds):
        """
        Remove the edges at the given indices, and return them as a dict, as `network.append()` takes them

        Args:
            inds (int, array, slice): the indices of the edges to remove
        """
        return self.get_inds(inds, remove=True)

    def remove_edges(self, inds):
        """
        Remove the specified edges from the network

        Args:
            inds (array): the indices of the edges to remove, or a boolean mask the same length as the network

        Example:
            ```python
            net = sim.networks.randomnet
            net.remove_edges(net.edges.p1 == 0) # Remove all edges with agent 0 as p1
            ```
        """
        inds = np.asarray(inds)
        if not inds.size: # Nothing to remove
            return
        if inds.dtype == bool:
            if len(inds) != len(self):
                errormsg = f'Boolean mask has length {len(inds)}, but the network has {len(self)} edges'
                raise ValueError(errormsg)
            keep = ~inds
        else:
            keep = np.ones(len(self), dtype=bool)
            keep[inds] = False
        for k in self.meta_keys():
            self.edges[k] = self.edges[k][keep]
        return

    def append(self, edges=None, **kwargs):
        """
        Add edges to the network

        Args:
            edges (dict): a dictionary of arrays with keys p1, p2, and optionally weight (1 if not given), as returned from network.pop_inds()
            kwargs (dict): the arrays of `edges`, by key
        """
        edges = sc.mergedicts(edges, kwargs)
        check_edges(edges)
        for key in self.meta_keys():
            curr_arr = self.edges[key]
            if key in edges:
                new_arr = edges[key]
            elif key in self.edge_defaults:
                new_arr = np.full(len(edges['p1']), self.edge_defaults[key], dtype=self.meta[key])
            else:
                errormsg = f'Cannot append edges since required key "{key}" is missing'
                raise KeyError(errormsg)
            self.edges[key] = np.concatenate([curr_arr, new_arr])
        self.validate_uids()
        return

    def record_flows(self):
        """ Store the number of edges in the network of the step (made in its step, so recorded after it) """
        super().record_flows()
        self.results['n_edges'][self.ti] = len(self)
        return

    def to_graph(self, max_edges=None, random=False): # pragma: no cover
        """
        The network as a NetworkX `DiGraph`, e.g. to draw it or to use NetworkX's measures of networks

        Args:
            max_edges (int): the maximum number of edges to show
            random (bool): if true, select edges randomly; otherwise, show the first N

        Examples:
            ```python
            import networkx as nx
            sim = ss.Sim(n_agents=100, networks='random').init()
            G = sim.networks.randomnet.to_graph()
            nx.draw(G)
            ```
        """
        import networkx as nx # Lazy import since slow
        keys = [('p1', int), ('p2', int), ('weight', float)]
        data = [np.array(self.edges[k], dtype=dtype) for k,dtype in keys]
        if max_edges:
            if random:
                inds = np.sort(np.random.choice(len(data[0]), max_edges, replace=False))
                data = [col[inds] for col in data]
            else:
                data = [col[:max_edges] for col in data]
        G = nx.DiGraph()
        G.add_weighted_edges_from(zip(*data), weight='weight')
        nx.set_edge_attributes(G, self.label, name='layer')
        return G

    def to_edgelist(self):
        """ The edges, as a list of (p1, p2) pairs """
        return list(zip(self.p1, self.p2))

    def to_dict(self):
        """ The edges, as a dict of arrays by column """
        return {k: self.edges[k] for k in self.meta_keys()}

    def to_df(self):
        """ The edges, as a dataframe """
        return sc.dataframe.from_dict(self.to_dict())

    def from_df(self, df, keys=None):
        """
        Set the edges from a dataframe or a dict of arrays

        Replaces the existing edges, and returns the network, so it can also be used to
        create a new network. If `weight` is not supplied, it defaults to 1.

        Args:
            df (dataframe/dict): the edges, with columns `p1`, `p2`, and (optionally) `weight`
            keys (list): the columns to use (default: the network's meta keys)

        Examples:
            ```python
            net = ss.Network(name='mynet').from_df(dict(p1=[0,1,2], p2=[1,2,3])) # Create a new network
            net.from_df(df) # Replace the edges of an existing network
            ```
        """
        check_edges(df)
        if keys is None:
            keys = self.meta_keys()
        for key in keys:
            if key in self.edge_defaults and key not in df:
                self.edges[key] = np.full(len(df['p1']), self.edge_defaults[key], dtype=self.meta.get(key))
            else:
                self.edges[key] = np.array(df[key], dtype=self.meta.get(key))
        self.validate_uids()
        return self

    def shrink(self):
        """ Remove the edges, to save memory (e.g. before saving the sim) """
        super().shrink()
        shrunk = ss.utils.shrink()
        self.edges = shrunk
        self.participant = shrunk
        self.agent_edges = None
        return

    def plot(self, max_edges=500, random=False, alpha=0.2, **kwargs):
        """
        Draw the network (or some of its edges), using NetworkX

        Args:
            max_edges (int): the maximum number of edges to show
            random (bool): if true, select edges randomly; otherwise, show the first N
            alpha (float): the alpha value of the edges
            kwargs (dict): figure and style arguments (e.g. `figsize`, `font`) are passed to `ss.plot_args()`, the rest to `nx.draw_networkx()`
        """
        import networkx as nx # Lazy import since slow
        figkeys = ss.utils.plotting_kw.fig + ss.utils.plotting_kw.style
        kw = ss.plot_args({k:kwargs.pop(k) for k in list(kwargs) if k in figkeys})
        with ss.style(**kw.style):
            fig,ax = plt.subplots(**kw.fig)
            G = self.to_graph(max_edges=max_edges, random=random)
            nx.draw_networkx(G, alpha=alpha, ax=ax, **kwargs)
            if max_edges:
                edgestr = f'{max_edges:n} of {len(self):n} connections shown' if len(self) > max_edges else f'{len(self):n} connections'
                plt.title(f'{self.label}: {edgestr}')
        return ss.return_fig(fig)

    def find_contacts(self, inds, as_array=True):
        """
        Find all contacts of the given people, e.g. for contact tracing

        Both ends of each edge are checked, and each contact is returned once, however many
        edges join them to the people given. So it is not for uses where several edges count
        for more than one, e.g. exposure risk.

        Args:
            inds (array): indices of people whose edges to return
            as_array (bool): if true, return as sorted array (otherwise, return as unsorted set)

        Returns:
            contact_inds (array/set): the indices of the pairing partners

        Example: If there were a network with
        - p1 = [1,2,3,4]
        - p2 = [2,3,1,4]
        Then find_contacts([1,3]) would return the array [1,2,3] (or the set {1,2,3}, with `as_array=False`)
        """
        lists = self.edge_index()
        if lists is None:
            return find_contacts(self.edges.p1, self.edges.p2, sc.toarray(inds), as_array=as_array)
        inds = np.asarray(sc.toarray(inds), dtype=np.int64)
        s1, e1, o1, s2, e2, o2 = lists
        partners = ss.solvers.backends.passes.partners(inds, s1, o1, s2, o2, len(s1) - 1)
        return partners if as_array else set(partners.tolist())

    def add_pairs(self):
        """ Add new edges: nothing in this base class, whose edges are given; subclasses say how pairs are formed """
        pass

    def remove_uids(self, uids):
        """
        Remove the edges of the given people, e.g. when they die (called by `People.remove_dead()`)

        Args:
            uids (array): the agents to remove
        """
        p1, p2 = self.edges.p1, self.edges.p2
        n = len(self.sim.people.alive.raw) if self.initialized else max(np.max(p1, initial=-1), np.max(p2, initial=-1)) + 1 # Every UID in the network
        is_removed = np.zeros(max(n, np.max(uids, initial=-1) + 1), dtype=bool) # A lookup table of the UIDs to remove (much faster than np.isin() for many edges)
        is_removed[uids] = True
        self.drop_edges(*ss.solvers.backends.passes.touching(p1, p2, is_removed))
        return

    def drop_edges(self, k, gone):
        """ Remove the `k` edges at the indices `gone` (in order; all edges if `k` is the number of edges), as `ss.solvers.backends.passes.ending()` gives them; not for the user """
        if not k:
            return
        for key in self.meta_keys():
            arr = self.edges[key]
            self.edges[key] = arr[:0] if k == len(arr) else ss.solvers.backends.passes.compress(arr, gone).view(type(arr))
        return

    def net_beta(self, disease_beta=None, inds=None, disease=None):
        """
        The probability that each edge transmits in a step, given the disease's probability (in the step, or on a network with acts, per act), for people with `rel_trans` and `rel_sus` of 1

        The edge's weight, and on a network with acts its number of acts in the step, scale the hazard: the probability is 1 − (1 − beta)^(weight·acts). The `rel_trans` and `rel_sus` of its two people, and the modifiers on the route, then scale the hazard this gives (see `Network.transmit()`). A network written as code can override this, for a disease written as code; a declared disease works out a network with acts from its hazard per act (see `ss.modify()` for what changes it on one route).

        Args:
            disease_beta (float): the disease's probability
            inds (array): the edges (default: all of them)
            disease (Disease): the disease
        """
        with np.errstate(divide='ignore', invalid='ignore'): # A chance of 1 is an infinite hazard, and stays 1 (but none on an edge of weight 0)
            return -np.expm1(np.log1p(-np.minimum(disease_beta, 1.0))*self._edge_times(inds)) # Beta should already include dt if desired

    def _edge_times(self, inds=None):
        """ How many times the disease's probability applies on each edge in a step: its weight, times its acts on a network with acts """
        if inds is None: inds = Ellipsis
        return self.edges.weight[inds]*self.edges.acts[inds] if self.has_acts else self.edges.weight[inds]

    def net_hazard(self, disease_beta=None, inds=None, disease=None):
        """
        The hazard of each edge in a step, −log(1 − `net_beta()`), which `rel_trans`, `rel_sus`, and the modifiers on the route then scale (see
        `ss.Network.transmit()`): worked out directly, with the same arguments, unless a subclass overrides `net_beta()`
        """
        with np.errstate(divide='ignore', invalid='ignore'): # A chance of 1 is an infinite hazard (but none on an edge of weight 0)
            if type(self).net_beta is not Network.net_beta:
                return -np.log1p(-self.net_beta(disease_beta=disease_beta, inds=inds, disease=disease))
            return -np.log1p(-np.minimum(disease_beta, 1.0))*self._edge_times(inds)

    def ends(self, d):
        """ The sources and targets of every edge, in direction `d`: p1 to p2 (0) or p2 to p1 (1) """
        return (self.edges.p1, self.edges.p2) if d == 0 else (self.edges.p2, self.edges.p1)


class DynamicNetwork(Network):
    """
    A network where partnerships update dynamically

    Args:
        memory (dur): how long it remembers partnerships that have ended (default: not at all); `find_contacts()`, and so
            contact tracing (`trace=` of a delivery), then also reaches the partners of the last `memory`, e.g. `memory=ss.months(12)`
            for partner notification of the past year's partners
        pars (dict): passed to `ss.Network`
        edges (dict/dataframe): passed to `ss.Network`
    """
    def __init__(self, pars=None, edges=None, memory=None, **kwargs):
        super().__init__(pars=pars, edges=edges, **kwargs)
        self.define_pars(memory=memory)
        self.meta.dur = ss_float # How many more steps each edge lasts
        self.history = None # With memory=: the partnerships that ended, as p1, p2, and the step they ended on
        return

    def step(self):
        """ Remove expired partnerships and add new ones """
        self.end_pairs()
        self.add_pairs()
        return

    def end_pairs(self):
        """ Remove partnerships whose duration has expired or whose members have died """
        n = len(self)
        k, gone = ss.solvers.backends.passes.ending(self.edges.dur, self.edges.p1, self.edges.p2, self.sim.people.alive.raw) # Those with no steps left after this one, or a partner not alive
        if self.pars.get('memory') is not None:
            ended = gone if k < n else slice(None)
            self.remember(self.edges.p1[ended], self.edges.p2[ended])
        self.drop_edges(k, gone)
        if k < n:
            self.edges.dur = self.edges.dur - 1 # dur is stored in units of self.t.dt (timesteps), so decrement by 1 per step
        return n

    def remember(self, p1, p2):
        """ With memory=: keep the partnerships that end now, and forget those that ended longer ago than it """
        m, sim, ti = self.pars.memory, self.sim, self.t.ti
        steps = (ss.solvers.convert(m, sim.unit_base) if isinstance(m, ss.TimePar) else m)/self.t.dt.to(sim.unit_base).value
        h = self.history if self.history is not None else sc.obj(p1=p1[:0], p2=p2[:0], ti=np.zeros(0, dtype=np.int32))
        keep = h.ti > ti - steps
        self.history = sc.obj(p1=np.concatenate([h.p1[keep], p1]), p2=np.concatenate([h.p2[keep], p2]), ti=np.concatenate([h.ti[keep], np.full(len(p1), ti, dtype=np.int32)]))
        return

    def find_contacts(self, inds, as_array=True):
        """ Find all contacts of the specified people (see `ss.Network.find_contacts()`), including, with memory=, their partners of the last `memory`

        Args:
            inds (array): the people
            as_array (bool): if True, return a sorted array rather than a set
        """
        if self.history is None:
            return super().find_contacts(inds, as_array=as_array)
        p1, p2 = np.concatenate([self.edges.p1, self.history.p1]), np.concatenate([self.edges.p2, self.history.p2])
        return find_contacts(p1, p2, sc.toarray(inds), as_array=as_array)


def check_acts(net, acts):
    """
    `acts=` of lasting partnerships is acts per unit time: a frequency (e.g. `ss.freqperweek(2)`, or a rate, `ss.perweek(2)`; a probability per unit time is an error), or a distribution of them drawn once for each
    partnership; a bare number, or a count per step drawn once, would depend on the step, so is an error
    """
    name = net.name or type(net).__name__.lower()
    example = 'e.g. ss.freqperweek(2), or ss.lognormal(ss.freqperyear(80), ss.freqperyear(30)) for a frequency that varies between partnerships'
    if acts is None or isinstance(acts, ss.Rate) and not isinstance(acts, ss.prob):
        return
    probs = [acts] if isinstance(acts, ss.prob) else []
    if isinstance(acts, ss.Dist):
        probs = [v for v in acts.pars.values() if isinstance(v, ss.prob)]
    if probs: # E.g. ss.probperyear(0.8), a chance, not a number of acts
        raise ValueError(f'acts= of "{name}" is a number of acts per unit time, not a probability per unit time (ss.{probs[0]!r}): give a frequency, {example}')
    elif sc.isnumber(acts):
        raise ValueError(f'acts= of "{name}" is a number of acts per unit time in each partnership; a bare number ({acts}) would be acts in a step, which depends on the step. Give a frequency, {example}')
    elif isinstance(acts, ss.Dist):
        timed = acts.unit is not None or any(isinstance(v, ss.TimePar) for v in acts.pars.values())
        if not timed:
            raise ValueError(f'acts= of "{name}" is a number of acts per unit time in each partnership, so a distribution of them needs its parameters as frequencies ({acts} has none), {example}')
        if isinstance(acts, (ss.poisson, ss.nbinom, ss.binom, ss.randint, ss.bernoulli)):
            raise ValueError(f'acts= of "{name}" is {acts}, a whole number of acts in a step, drawn once for each partnership, which depends on the step (at weekly steps, a fifth of partnerships would never have sex). Give a frequency, {example}')
        return
    raise TypeError(f'acts= of "{name}" is {acts} ({type(acts).__name__}); give a rate, {example}')


# %% Specific instances of networks

class RandomNet(DynamicNetwork):
    """
    Random connectivity between agents

    Each agent has a number of contacts (`n_contacts`), and the other end of each is
    drawn from the other agents' contacts (`method='fast'`, the default), so an agent's
    chance of being picked is proportional to its own number of contacts. This keeps
    the mixing structure of risk-stratified networks (a small high-risk core with many
    contacts forms most of its edges with itself).

    `method` says how the other end of each contact is drawn:

    - `'fast'` (the default): by sampling the contacts *with* replacement. The source
      side of each edge is exactly `n_contacts`, while the target side is
      Poisson-distributed with mean `n_contacts`, so agents have somewhat more variable
      numbers of contacts than with `'exact'`; about 1.8x faster than `'exact'`.
    - `'exact'`: by shuffling the contacts (sampling *without* replacement), so each
      agent has exactly `n_contacts` contacts, if it is even: each agent starts half of
      its contacts and is the other end of as many, so an odd number is rounded at random
      to the even numbers either side for each agent (5 gives 4 or 6, 5 on average).
    - `'uniform'`: uniformly from all eligible agents, regardless of their own
      `n_contacts`, as in Covasim v3's random networks. This gives less variable numbers
      of contacts (e.g. with `n_contacts=ss.poisson(20)`, a variance of about 15, rather
      than 30 for `'fast'` or 20 for `'exact'`), but loses the mixing structure above.
    - `'safe'`: by giving each contact a random place on a line, from the agent's own
      random numbers, and joining it to its nearest neighbor there. Adding an agent then
      changes only the contacts next to theirs, which suits scenario comparisons, but it
      is slower, and does not yet work within places.

    Note: not CRN safe when people are born or die, unless `dynamic=False`: with any of these
    methods but `'safe'`, one death rewires almost every edge.

    By default (`dur=0`), each contact lasts one step of the network: with `dynamic=True` (the
    default), the contacts are drawn again on every step, and with `dynamic=False`, once. With
    `dur=` above 0, each contact lasts that long, and the ones that end are replaced by new
    ones drawn at random. For partnerships formed toward a target number of partners, see `ss.DegreeNet`.

    Args:
        n_contacts (float/`ss.Dist`): the number of (bidirectional) contacts of each agent: a number is the same for everyone (rounded at random, if it is not whole, so it is the mean), and a distribution gives its spread (e.g. `ss.poisson(10)`)
        dur (float/`ss.Duration`/`ss.Dist`): how long each contact lasts: a duration (the same for every contact), or a distribution; a bare number is in the sim's unit (default: 0, one step)
        who (Cond/str/func): if given, who takes part: a condition over people (e.g. `(ss.person.age >= 18) & (ss.person.age < 65)`), as for `ss.DegreeNet`; nobody else is in the network
        dynamic (bool): if True (default), contacts are replaced once they end (with `dur=0`, every step); if False, they are drawn once and never change
            (so agents who join later, e.g. newborns, are not added)
        memory (dur): how long it remembers contacts that have ended, for contact tracing (see `ss.DynamicNetwork`)
        method (str): how the other end of each contact is drawn: 'fast' (the default), 'exact', 'uniform', or 'safe' (see above)
        remove_self_loops (bool): if True, drop the few edges that join an agent to itself (as any of the methods can make; e.g. about 1 in 25 edges with 50 agents, and far fewer in larger populations), so agents have slightly fewer contacts; off by default since such edges transmit nothing anyway. The same pair of agents can still be joined by more than one edge.
        pars (dict): the same parameters, as a dict

    Note: n_contacts = 10 will create *5* edges per agent. Since disease transmission
    usually occurs bidirectionally, this means that the effective number of contacts
    per agent is actually 10. Consider 3 agents with 3 edges between them (a triangle):
    each agent is connected to 2 other agents.

    Example:
        ```python
        # Static network with Poisson-distributed contacts among adults, as in Covasim v3
        net = ss.RandomNet(n_contacts=ss.poisson(20), who=(ss.person.age >= 18) & (ss.person.age < 65), dynamic=False, method='uniform')
        ```
    """
    methods = ['fast', 'exact', 'uniform', 'safe']

    def __init__(self, pars=None, n_contacts=_, dur=_, who=_, dynamic=_, method=_, remove_self_loops=_, memory=_, **kwargs):
        ss.v3_deprecations.check(type(self), '__init__', list(kwargs) + list(pars or {}))
        super().__init__()
        self.define_pars(
            n_contacts = ss.constant(10),
            dur = 0, # One step
            who = None,
            dynamic = True,
            method = 'fast',
            remove_self_loops = False,
        )
        self.update_pars(pars, **kwargs)
        if self.pars.method not in self.methods:
            errormsg = f'method= of ss.RandomNet() is one of {sc.strjoin(self.methods)}, not {self.pars.method!r}'
            raise ValueError(errormsg)
        self.dist = ss.Dist(distname='RandomNet') # Default RNG; the same name for every method, so results stay reproducible
        if self.pars.method == 'safe':
            self.position = ss.random(name='position') # Each contact's own place on a line, by agent
            self.position.hash_dtype = np.float64 # Since float32 only gives 2^24 distinct values, which would mean many ties
        return

    def mean_contacts(self, sim):
        """ Its mean number of contacts (see `ss.Network.mean_contacts()`)

        Args:
            sim (Sim): the sim being initialized
        """
        n_contacts = self.pars.get('n_contacts')
        if n_contacts is None:
            return super().mean_contacts(sim)
        return ss.solvers.Mixing.mean(n_contacts, sim.unit_base), 'its mean number of contacts'

    def get_edges(self, inds, n_contacts):
        """
        Find new edges, by `method=`

        Each person is at both ends of edges: a person given 5 contacts appears 5 times
        in `source` and, on average, 5 times in `target`, so has 10 in all. So `n_contacts`
        here is HALF the contacts a person is expected to have, when both are used (as for
        social contacts).

        Args:
            inds (list/array): person indices
            n_contacts (list/array): the same length as `inds` with the number of bidirectional contacts to assign to each person

        Returns:
            Two arrays, for source and target
        """
        if self.pars.method == 'exact':
            return self.shuffle_edges(inds, n_contacts)
        elif self.pars.method == 'safe':
            return self.nearest_edges(inds, n_contacts)
        nc = np.asarray(n_contacts)
        by_agent = self.pars.method == 'uniform' or nc.size <= 1 or (nc == nc.flat[0]).all() # With the same number of stubs each, the owner of a stub drawn at random is an agent drawn at random
        seed = self.dist.rng.integers(2**62) # The key of this step's uniforms, one for each edge (from a counter, so a large network can be made by several threads)
        source, target = ss.solvers.backends.passes.random_edges(inds, nc, seed, by_agent) # A random target stub for each source stub; sampling stubs rather than agents keeps selection probability proportional to n_contacts
        source, target = source.view(ss.uids), target.view(ss.uids)
        self.dist.jump() # Reset the RNG manually, as in shuffle_edges()
        return source, target

    def shuffle_edges(self, inds, n_contacts):
        """ Find edges by shuffling the contacts, so each agent is at each end of exactly as many as given (`method='exact'`), so has twice that many """
        source = np.repeat(inds, n_contacts)
        target = source.copy()
        randvals = self.dist.rng.random(len(target)) # Precompute the random values for the shuffle
        fisher_yates_shuffle(target, randvals) # Shuffle the target in place (faster equivalent of rng.permutation)
        self.dist.jump() # Reset the RNG manually; does not auto-jump since using rng directly above # TODO, think if there's a better way
        return source, target

    def nearest_edges(self, inds, n_contacts):
        """ Find edges by giving each contact a random place on a line, and joining it to its nearest neighbor there (`method='safe'`) """
        nc = np.broadcast_to(n_contacts, np.shape(inds))
        agents = [inds[nc > i] for i in range(nc.max(initial=0))] # Necessary loop, since each draw is by agent
        if not len(agents):
            return ss.uids(), ss.uids()
        place = np.concatenate([self.position.rvs(these) for these in agents])
        agent = np.concatenate(agents)
        order = np.argsort(place)
        place, agent = place[order], agent[order]
        center = place[1:-1]
        use_prev = np.abs(center - place[:-2]) <= np.abs(center - place[2:]) # Ties go to the lower neighbor
        return agent[1:-1].view(ss.uids), np.where(use_prev, agent[:-2], agent[2:]).view(ss.uids)

    @staticmethod
    def draw_places(share, home, u):
        """ The place of each contact, from the time share between places (a sparse array, by rows) of the home of each, and a uniform number for each: the first place where the cumulative share reaches it """
        out = np.full(len(home), share.shape[1] - 1)
        order = np.argsort(home, kind='stable')
        counts = np.bincount(home, minlength=share.shape[0])
        ends = np.cumsum(counts)
        for r in np.flatnonzero(counts):
            these = order[ends[r]-counts[r]:ends[r]]
            cols, cum = share.indices[share.indptr[r]:share.indptr[r+1]], np.cumsum(share.data[share.indptr[r]:share.indptr[r+1]])
            k = np.searchsorted(cum, u[these], side='left')
            found = k < len(cols)
            out[these[found]] = cols[k[found]]
        out[u <= 0] = 0 # As the cumulative share of the first place, which can be 0
        return out

    def pair(self, inds, n_contacts):
        """
        Find edges (see `get_edges()`); in a model with places, within each place, as a compartmental run takes it to be

        With commuting or visiting between places, each of a person's contacts is in a
        place drawn from the share of their time they spend in each.
        """
        mixing = getattr(self.sim, 'mixing', None)
        if mixing is None or mixing.space is None:
            return self.get_edges(inds, n_contacts)
        if self.pars.method == 'safe':
            errormsg = f"method='safe' of ss.RandomNet() does not yet work within places: use one of {sc.strjoin(self.methods[:-1])}"
            raise NotImplementedError(errormsg)
        stubs = np.repeat(inds, n_contacts)
        place = self.sim.groups.level_raw(mixing.space)[stubs]
        mobility = self.sim.groups.strats[mixing.space].mobility
        if mobility is not None and mobility.kind in ['commute', 'visit']:
            place = self.draw_places(mixing.time_share, place, self.dist.rng.random(len(stubs)))
            self.dist.jump()
        return self.pair_within(stubs, place, self.sim.groups.strats[mixing.space].n, inds, n_contacts)

    def pair_within(self, stubs, place, n_places, inds, n_contacts):
        """ Find edges among the contacts (`stubs`) in each of `n_places` places, the place of each given by `place` """
        order = np.argsort(place, kind='stable')
        ends = np.cumsum(np.bincount(place, minlength=n_places))
        p1, p2 = [], []
        for a,b in zip(np.append(0, ends[:-1]), ends):
            if b > a:
                src, trg = self.get_edges(stubs[order[a:b]], np.ones(b - a, dtype=int))
                p1.append(src)
                p2.append(trg)
        if not p1:
            return self.get_edges(inds, n_contacts)
        return np.concatenate(p1).view(ss.uids), np.concatenate(p2).view(ss.uids)

    @property
    def one_step(self):
        """ With `dur=0` (the default), and stepping with the sim, every edge ends at the next step, before anything transmits over it, so the dead's edges need not be removed before then (see `ss.networks.Route.one_step`) """
        p = self.pars
        return p.dynamic and p.get('memory') is None and not isinstance(p.dur, ss.Dist) and p.dur == 0 and self.on_sim_time

    @property
    def on_sim_time(self):
        """ Whether the network steps with the sim (worked out once) """
        if getattr(self, '_on_sim_time', None) is None:
            self._on_sim_time = self.sim is not None and self.sim.loop._on_sim_time(self)
        return self._on_sim_time

    def step(self):
        """ If dynamic, replace expired edges; otherwise, do nothing """
        if self.pars.dynamic:
            super().step()
        return

    def same_contacts(self, n, current):
        """
        The number of new contacts for each of `n` agents when it is the same whole number for everyone, so that nothing need be drawn: when
        `n_contacts` is a whole number, and the edges needed (with `current` edges) give each a whole number (as every step with `dur=0`); else
        None. Only with common random numbers, since without them a draw not made would change the draws after it.
        """
        nc = self.pars.n_contacts
        v = nc.pars.value if isinstance(nc, ss.constant) and nc.unit is None else None
        if not ss.options.crn or not sc.isnumber(v) or not float(v).is_integer() or not n or not v:
            return None
        orig = v*n
        needed = orig/2 - current
        scaled = v*needed/orig # The same arithmetic as add_pairs(), for one agent
        return int(scaled) if needed > 0 and float(scaled).is_integer() else None

    def edge_steps(self, p1):
        """ How many steps each new edge lasts after this one, from `dur=`: drawn for each if a distribution; a bare number is in the sim's unit """
        dur = self.pars.dur
        if isinstance(dur, ss.Dist):
            return dur.rvs(p1)
        n = len(p1)
        if dur == 0:
            return np.zeros(n, dtype=ss_float) # In its dtype (half the memory of float64), as validate() would make it
        unit = self.sim.unit_base
        dur = ss.solvers.convert(dur, unit) if isinstance(dur, ss.TimePar) else dur
        return np.full(n, dur/self.t.dt.to(unit).value, dtype=ss_float)

    def add_pairs(self):
        """ Add the new edges needed to bring each agent up to their number of contacts """
        p = self.pars
        people = self.sim.people
        uids = self.eligible(p.who).uids
        current = len(self)
        n_conn = self.same_contacts(len(uids), current)
        if n_conn is None:
            if isinstance(p.n_contacts, ss.Dist):
                n_conn = p.n_contacts.rvs(uids)
            else:
                n_conn = np.full(len(people), p.n_contacts)

            # See how many new edges we need
            orig = n_conn.sum()
            target = orig / 2 # Divide by 2 since bi-directional
            needed = target - current
            if needed <= 0:
                return
            n_conn = n_conn*needed/orig
            n_conn = p.n_contacts.randround(n_conn) # CRN-safe stochastic integer rounding

        p1, p2 = self.pair(uids, n_conn)
        if p.remove_self_loops:
            keep = p1 != p2 # One pass over the new edges; finding duplicates would need a sort of all of them
            p1, p2 = p1[keep], p2[keep]
        n = len(p1)
        weight = np.ones(n, dtype=ss_float)
        dur = self.edge_steps(p1)

        p1 = p1.view(ss.uids) # No copy
        p2 = p2.view(ss.uids)
        if current == 0: # Nothing to append to
            self.edges.p1 = p1
            self.edges.p2 = p2
            self.edges.weight = weight
            self.edges.dur = dur
        else:
            self.append(p1=p1, p2=p2, weight=weight, dur=dur)
        return


class DegreeNet(DynamicNetwork):
    """
    Partnerships that form and dissolve, with a target number of partners at a time

    The number of partners people have at any moment (`n_contacts`) is a target
    for the whole population, not a fixed number for each person: partnerships end
    at random, and people look for new ones at a rate that depends on how many
    they already have, chosen so that the distribution of the number of partners
    stays at the target. Everyone moves through it over time, as in EpiModel's
    networks fitted to target statistics: this covers the same ground as a dyad-independent
    exponential random graph model (ERGM), with the terms for edges, degree, and mixing, but not
    terms such as triangles. The network starts at the target. In an
    open population, partnerships that end at a constant rate (`dur` a mean) end less
    often by as many as are ended by a partner leaving (dying), as EpiModel's d.rate
    correction does, so they last as long as given on average; with any other
    distribution of durations, people look for partners more (or less) often, by a
    factor that keeps the mean number of partners at the target. Partnerships are
    not only sexual: people who share needles have partners too. For sexual
    partnerships, with an age of debut, see `ss.SexualNet`.

    Each partnership can be between one person of each of two groups (`between`), e.g. a
    man and a woman, so a disease's beta on it can differ by direction; each group can then
    have its own number of partners (`n_contacts` as a dict). With no target (the
    default), people who have no partner are paired on every step, so nobody has more
    than one at a time.

    A subclass that overrides `step()` forms its own partnerships: the base starts none for
    it, and has none of the parameters for forming them, only `between`.

    Note: not CRN safe when people are born or die.

    Args:
        n_contacts (float/`ss.Dist`/`ss.by`/dict): if given, a target for the number of partners a person has at a time: a number, which is the mean of a Poisson (whose shape `concurrency` and `max_contacts` can change), or a distribution over whole numbers (e.g. `ss.constant(1)`, for exactly one); or a number for each level of a property of people (`ss.by(risk=[0.4, 1.6])`), as EpiModel's nodefactor; or one for each group of `between`, e.g. `dict(fsw=3, client=0.5)` (each partnership has one of each, so these should give the two groups about as many partners in all). By default, no target: at most one partner, pairing people who have none on every step
        dur (float/`ss.Duration`/`ss.Dist`): how long partnerships last: a mean (they end at a constant rate), or a distribution; a bare number is in the sim's unit (default: 10)
        acts (rate/`ss.Dist`): if given, the number of acts per unit time in each partnership: a frequency such as `ss.freqperweek(2)` (or a rate, `ss.perweek(2)`), or a distribution of them drawn once for each partnership; a disease's beta is then per act (see `ss.peract`)
        who (Cond/str/func): if given, who takes part: a condition over people (e.g. `ss.person.pwid`), as for `ss.Event()`; nobody else is in the network
        between (str/tuple): if given, who partnerships are between: `'mf'`, a man and a woman, so its directions are `m2f` and `f2m` and a disease's beta on it is `ss.peract(m2f=..., f2m=...)`, or one value for both; `'mm'`, two men, with one beta; or a pair of groups, each the name of a boolean property of people or of a level of a stratification, e.g. `('fsw', 'client')`, whose directions are `fsw2client` and `client2fsw` (nobody can be in both)
        concurrency (float): if given, the share of people with two or more partners at once; the distribution of `n_contacts` is adjusted to give it, keeping its mean
        max_contacts (int): if given, nobody has more partners than this at once
        match (dict): if given, the share of partnerships between people at the same level of a property of people (a stratification, or an attribute such as `'female'`), e.g. `dict(risk=0.8)`, as EpiModel's nodematch; or of each of several, e.g. `dict(risk=0.8, region=0.6)`
        memory (dur): how long it remembers partnerships that have ended, for contact tracing (see `ss.DynamicNetwork`)
        pars (dict): the same parameters, as a dict

    The realized network is recorded in the results (`mean_degree`, `concurrency`, and
    `match`, or with several properties `match_<name>` for each), and `net.check()` compares it with the targets.

    Examples:
        ```python
        # EpiModel's introductory network: 175 partnerships among 500 people, 110 people with two or more partners, nobody with four or more, and partnerships lasting 50 steps
        net = ss.DegreeNet(n_contacts=0.7, concurrency=0.22, max_contacts=3, dur=50)

        # Most partnerships within a risk group, and over half within a region
        net = ss.DegreeNet(n_contacts=0.7, dur=50, match=dict(risk=0.8, region=0.6))

        # And four times as many partners in the high-risk group (with sim.stratify(risk=['low', 'high']))
        net = ss.DegreeNet(n_contacts=ss.by(risk=[0.4, 1.6]), dur=50, match=dict(risk=0.8))

        # Needle sharing among people who inject drugs (a boolean property of people), with 20 injections a month in each partnership
        needles = ss.DegreeNet(name='needles', who=ss.person.pwid, n_contacts=2, dur=ss.months(6), acts=ss.freqpermonth(20))

        # Sex work, between female sex workers and their clients (boolean properties of people)
        sw = ss.DegreeNet(name='sw', between=('fsw', 'client'), n_contacts=dict(fsw=5, client=0.5), dur=ss.months(1), acts=ss.freqperweek(1))
        ```
    """
    def __init__(self, pars=None, n_contacts=_, dur=_, concurrency=_, max_contacts=_, match=_, acts=_, who=_, between=_, memory=_, **kwargs):
        ss.v3_deprecations.check(type(self), '__init__', list(kwargs) + list(pars or {}))
        super().__init__()
        self.define_pars(between=None)
        if not self.forms_own: # A subclass that forms its own partnerships uses none of these
            self.define_pars(
                n_contacts = None, # No target: at most one partner
                dur = 10,
                concurrency = None,
                max_contacts = None,
                match = None,
                acts = None,
                who = None,
            )
        self.update_pars(pars, **kwargs)
        self.set_between()
        if self.pars.get('acts') is not None:
            self.meta.acts = ss_float # Acts in each partnership in a step: its acts per unit time times the step
        self.rng_degree = ss.random() # The number of partners each person starts with
        self.rng_order  = ss.random() # The order in which people are paired
        self.rng_seek   = ss.random() # Whether each person looks for a partner on this step
        self.rng_end    = ss.random() # When each partnership ends, if it ends at a constant rate
        self.rng_stubs  = ss.random() # For shuffling the partners people start with
        self.rng_match  = ss.random() # Whether each person looks for a partner at their own level, with match=
        self.departures = 0.0 # The share of partnerships ended each step by a partner leaving (dying), estimated as the run goes
        self.lost = 0 # Partnerships ended that way on this step
        self.waiting = ss.uids() # People who looked for a partner and were left over, who look again
        return

    def agent_reasons(self, sim):
        """ Lasting partnerships are approximate in compartments, so `solver='auto'` keeps agents """
        return [ss.modules.Preference(f'"{self.name}" ({type(self).__name__}) has lasting partnerships, which agents follow exactly')]

    def set_between(self):
        """ The groups of `between` (or None), and from them the sides of the network """
        between = self.pars.between
        if between is None:
            self.groups = None
            return
        groups = dict(mf=('male', 'female'), mm=('male', 'male')).get(between) if isinstance(between, str) else between
        if not isinstance(groups, (tuple, list)) or len(groups) != 2 or not all(isinstance(g, str) for g in groups):
            errormsg = f'between= of "{self.name}" is "mf" (men and women), "mm" (men who have sex with men), or a pair of groups of people by name, e.g. ("fsw", "client"), not {between!r}'
            raise ValueError(errormsg)
        self.groups = tuple(groups)
        self.sides = None if groups[0] == groups[1] else {dict(male='m', female='f').get(g, g):g for g in groups} # Its directions: m2f and f2m between men and women; within one group, one beta
        return

    @property
    def forms_own(self):
        """ Whether a subclass forms its own partnerships (it overrides `step()`), so the base forms none for it """
        return type(self).step is not DegreeNet.step

    @property
    def targeted(self):
        """ Whether the number of partners people have at a time is a target (`n_contacts`), rather than one at most """
        return self.pars.get('n_contacts') is not None

    def side_raw(self):
        """ The group of `between` each person is in (-1 for neither), indexed by UID: by sex, men 0 and women 1 """
        if self.groups == ('male', 'female'):
            return self.sim.people.female.raw.astype(np.int8)
        elif self.groups is not None and self.sides is None:
            return np.where(people_mask(self.sim, self.groups[0]), 0, -1).astype(np.int8)
        return super().side_raw()

    def active(self, people):
        """ The agents who can have partners, as a boolean array: alive, and with `who=`, those it gives, and with `between`, those in one of its groups

        Args:
            people (People): the people
        """
        active = people.alive
        if self.pars.get('who') is not None:
            active = active & people_mask(self.sim, self.pars.who)
        if self.groups == ('male', 'male'):
            return active & people.male
        elif self.groups is not None and self.groups != ('male', 'female'): # Otherwise everyone is in one
            return active & (self.side_raw() >= 0)
        return active

    def init_pre(self, sim):
        """ Check the parameters, and work out the targets: the distribution of the number of partners, and how often people look for one

        Args:
            sim (Sim): the sim being initialized
        """
        match = self.pars.get('match')
        if match is not None and (not isinstance(match, dict) or not match or not all(sc.isnumber(v) and 0 <= v <= 1 for v in match.values())):
            errormsg = f'match= of "{self.name}" takes properties of people and the share of partnerships at the same level of each, e.g. dict(risk=0.8), not {match}'
            raise ValueError(errormsg)
        for key in list(match or {})[1:]: # Whether each person looks within their own level of each other property, with several
            setattr(self, f'rng_match_{key}', ss.random())
        self.match_extra = np.zeros(len(match or {})) # For each property, how many more look within their level than by the target alone: see match_pairs()
        self.match_results = {key:'match' if len(match) == 1 else f'match_{key}' for key in match or {}} # The result of the share at the same level of each property
        super().init_pre(sim)
        if self.forms_own:
            return
        p = self.pars
        check_acts(self, p.acts)
        if isinstance(p.acts, ss.TimePar):
            self.acts_per_step = ss.solvers.convert(p.acts, sim.unit_base)*self.t.dt.to(sim.unit_base).value
        if isinstance(p.dur, ss.Dist):
            self.mean_dur = None # Only for partnerships that end at a constant rate
            mean = ss.transitions.dwell_stats(p.dur, lambda v: ss.solvers.convert(v, sim.unit_base))[0]
            self.mean_steps = mean/self.t.dt.to(sim.unit_base).value
        else:
            dur = ss.solvers.convert(p.dur, sim.unit_base) if isinstance(p.dur, ss.TimePar) else p.dur
            self.mean_dur = self.mean_steps = dur/self.t.dt.to(sim.unit_base).value # In steps
            if self.mean_dur < 1:
                errormsg = f'Partnerships of "{self.name}" last {self.mean_dur:n} steps on average; they need to last at least one'
                raise ValueError(errormsg)
        if not self.targeted:
            if p.concurrency is not None or p.max_contacts is not None:
                errormsg = f'concurrency= and max_contacts= of "{self.name}" are targets for the number of partners at a time, so need n_contacts= as well'
                raise ValueError(errormsg)
            return
        self.by_level = None # With n_contacts=ss.by(...): the property of people whose levels have degrees of their own
        if isinstance(p.n_contacts, ss.by):
            if len(p.n_contacts.dims) != 1 or p.n_contacts.dim is None:
                errormsg = f'n_contacts= of "{self.name}" takes a number for each level of one property of people, e.g. ss.by(risk=[0.4, 1.6]), not {p.n_contacts}'
                raise ValueError(errormsg)
            self.by_level = p.n_contacts.dim
            values = sc.tolist(p.n_contacts.values)
        elif isinstance(p.n_contacts, dict):
            if self.sides is None or set(p.n_contacts) != set(self.groups):
                errormsg = f'n_contacts= of "{self.name}" as a dict gives a value for each group of between= ({sc.strjoin(list(dict.fromkeys(self.groups or [])))}), not for {sc.strjoin(list(p.n_contacts))}'
                raise ValueError(errormsg)
            values = [p.n_contacts[g] for g in self.groups]
        else:
            values = [p.n_contacts]
        dists = [self.degree_dist(v, p.concurrency, p.max_contacts) for v in values]
        size = max(len(d) for d in dists)
        self.dists_degree = np.array([np.pad(d, (0, size - len(d))) for d in dists]) # One row for each level or group, with n_contacts by level or group
        self.dist_degree = self.dists_degree.mean(axis=0) # For everyone, for the results and check(): the mix of the rows, from init_post()
        self.boost = np.ones(len(dists)) # With durations not at a constant rate, how much more (or less) often people look than in a closed population, adjusted as the run goes: see adjust()
        self.init_seek(self.mean_steps)
        if p.match is not None and self.by_level is None: # Who partners whom, but not how many: a common slip when porting a model with more partners at some levels (EpiModel's degree by risk)
            key = list(p.match.keys())[0]
            ss.note(f'"{self.name}" matches partners by {key}, but everyone has the same number of partners whatever their {key}; for more at some levels, give n_contacts=ss.by({key}=[...])', sim=self.sim)
        return

    @staticmethod
    def degree_dist(n_contacts, concurrency=None, max_contacts=None):
        """ The target share of people with 0, 1, 2, ... partners at a time """
        if sc.isnumber(n_contacts):
            n_contacts = ss.poisson(n_contacts)
        pars = {k:v for k,v in n_contacts.pars.items()}
        top = max_contacts
        if isinstance(n_contacts, ss.poisson):
            mean = pars['mean']
            k = np.arange(sc.ifelse(top, int(mean + 10*np.sqrt(mean) + 10)) + 1)
            base = ss.dists.sps.poisson.pmf(k, mean)
        elif isinstance(n_contacts, ss.constant): # A whole number for everyone, or the two either side of it, as for a number rounded at random (0.7 is 1 for 70% of people)
            mean = float(pars['value'])
            lo = int(np.floor(mean))
            k = np.arange(sc.ifelse(top, lo + 1) + 1)
            base = (k == lo)*(1 - (mean - lo)) + (k == lo + 1)*(mean - lo)
        else: # Any other distribution, by sampling a copy of it
            draws = np.round(sc.dcp(n_contacts).init(force=True).rvs(100_000)).astype(int)
            mean = draws.mean()
            k = np.arange(sc.ifelse(top, draws.max()) + 1)
            base = np.bincount(np.clip(draws, 0, k[-1]), minlength=len(k)).astype(float)
        if concurrency is None and max_contacts is None:
            return base/base.sum()

        import scipy.optimize as spo # Imported here, since only this needs it
        def tilt(x): # Reweight the distribution exponentially in the number of partners and in having two or more
            w = base*np.exp(x[0]*k + (x[1] if len(x) > 1 else 0)*(k >= 2))
            return w/w.sum()
        if concurrency is None:
            x = [spo.brentq(lambda a: tilt([a]) @ k - mean, -50, 50)]
        else:
            x = spo.fsolve(lambda x: [tilt(x) @ k - mean, tilt(x)[2:].sum() - concurrency], [0, 0])
        out = tilt(x)
        if abs(out @ k - mean) > 1e-6 or (concurrency is not None and abs(out[2:].sum() - concurrency) > 1e-6):
            hint = '; ss.constant() has no spread for concurrency= to reshape: give a number (the mean of a Poisson) or another distribution' if isinstance(n_contacts, ss.constant) else ''
            errormsg = f'No distribution of the number of partners has a mean of {mean:n}{f", {concurrency:n} with two or more," if concurrency is not None else ""} and a maximum of {max_contacts}{hint}'
            raise ValueError(errormsg)
        return out

    @staticmethod
    def seek_probs(dist, mean_dur):
        """ The probability that someone with each number of partners looks for another on a step, for a target distribution, given how many steps partnerships last on average """
        k = np.arange(len(dist))
        ratio = np.divide(dist[1:], dist[:-1], out=np.where(dist[1:] > 0, np.inf, 0.0), where=dist[:-1] > 0) # Below the least number in the target (e.g. none with ss.constant(1)), people always look
        return np.minimum(np.append((k[:-1] + 1)*(1/mean_dur)*ratio, 0), 1) # Chosen so the target is the stationary distribution: those with k partners who gain one balance those with k+1 who lose one

    def init_seek(self, mean_steps):
        """ The probability that someone in each row of the targets with each number of partners looks for another on a step, given how many steps partnerships last on average (for each row) """
        self.p_seeks = np.array([self.seek_probs(d, m) for d,m in zip(self.dists_degree, np.broadcast_to(mean_steps, len(self.dists_degree)))])
        return

    def rows(self, uids):
        """ The row of each person in the targets: their level, with n_contacts by level, or their group, with n_contacts by group of between (else 0 for all) """
        if self.by_level is not None:
            return np.asarray(self.levels(self.by_level))[uids].astype(int)
        return self.side_raw()[uids] if len(self.dists_degree) > 1 else np.zeros(len(uids), dtype=int)

    def seek_prob(self, uids):
        """ The probability that each person looks for a partner on this step, given how many they have """
        return self.p_seeks[self.rows(uids), np.minimum(self.degrees(uids), self.p_seeks.shape[1] - 1)]

    def degrees(self, uids):
        """ The number of partners each person has """
        size = int(uids.max()) + 1 if len(uids) else 0
        counts = np.bincount(self.edges.p1, minlength=size)[:size] + np.bincount(self.edges.p2, minlength=size)[:size]
        return counts[uids]

    def levels(self, name=None):
        """ The level of every person in a property of people (by default, the one match= names), as an array indexed by UID """
        sim = self.sim
        name = list(self.pars.match.keys())[0] if name is None else name
        names = [strat.name for strat in sim.groups.strats]
        if name in names: # A stratification
            return sim.groups.level_raw(names.index(name))
        return np.asarray(sim.people[name].raw)

    def init_results(self):
        """ With a target, the partnerships formed and the realized number of partners; with match=, the share at the same level """
        super().init_results()
        if self.targeted:
            self.define_results(
                ss.Result('formed', label='Partnerships formed', kind='flow', auto_plot=False),
                ss.Result('mean_degree', label='Mean number of partners', kind='ratio', scale=False, auto_plot=False), # Per person, so not scaled by the population
                ss.Result('concurrency', label='Share with two or more partners', kind='ratio', scale=False, auto_plot=False),
            )
        for key,name in getattr(self, 'match_results', {}).items():
            self.define_results(ss.Result(name, label=f'Share of partnerships at the same level{"" if name == "match" else f" of {key}"}', kind='ratio', scale=False, auto_plot=False))
        return

    def match_pairs(self, people, rvs=None):
        """
        Pair people who are looking for partners, who are in a random order: in turn, one of each group of `between`
        (or two of one), or, with `match=`, some within their own level, so that the share of partnerships at the
        same level is the target

        With several properties, each person looks within their own level of each with its own chance, as with one,
        and is paired with someone who looks within the same levels of the same properties. This gives each target
        when the properties are independent of each other; `net.check()` says how close it is otherwise.

        Args:
            people (uids): the people, once for each partner they look for
            rvs (array): a random number for each, for each property (properties by people), which says whether they look within their level (by default, drawn for each person)

        Returns the two lists of partners, and anyone left over.
        """
        side = self.side_raw()[people] if self.sides is not None else None

        def split(idx): # Partners in turn, by position, and anyone left over
            if side is None:
                n = len(idx)//2
                return idx[:n], idx[n:2*n], idx[2*n:]
            a, b = idx[side[idx] == 0], idx[side[idx] == 1]
            n = min(len(a), len(b))
            return a[:n], b[:n], np.concatenate([a[n:], b[n:]])

        if self.pars.match is None: # In turn
            p1, p2, left = split(np.arange(len(people)))
            return people[p1], people[p2], people[left]
        if rvs is None:
            rvs = [self.rng_match.rvs(people)] + [getattr(self, f'rng_match_{key}').rvs(people) for key in list(self.pars.match)[1:]]
        codes, sizes, levels = [], [], [] # The level each person looks within for each property (0 for none), how many there can be, and each person's level
        s = side if side is not None else np.zeros(len(people), dtype=np.int8)
        for (key,share),u,extra in zip(self.pars.match.items(), np.reshape(rvs, (len(self.pars.match), -1)), self.match_extra):
            level = np.unique(self.levels(key)[people], return_inverse=True)[1]
            frac = [np.bincount(level[s == g], minlength=level.max(initial=0) + 1)/max(np.sum(s == g), 1) for g in [0, 1 if side is not None else 0]]
            by_chance = frac[0] @ frac[1] # The share at the same level if they were paired at random
            within = np.clip((share - by_chance)/(1 - by_chance) + extra, 0, 1) if by_chance < 1 else 1.0
            codes.append(np.where(u < within, level + 1, 0))
            levels.append(level)
            sizes.append(len(frac[0]) + 1)
        p1, p2 = [], []
        left = np.arange(len(people))
        K = len(codes)
        for keep in [ks for n in range(K, 0, -1) for ks in itertools.combinations(range(K), n)]: # Within the levels of every property each looks within; anyone left over, within those of fewer, and so on
            cell = np.zeros(len(left), dtype=np.int64)
            for k in keep:
                cell = cell*sizes[k] + codes[k][left]
            rest = [left[cell == 0]]
            for c in np.unique(cell[cell > 0]):
                a, b, other = split(left[cell == c])
                p1.append(a)
                p2.append(b)
                rest.append(other)
            left = np.concatenate(rest)
        a, b, left = split(left) # The rest at random
        p1, p2 = np.concatenate(p1 + [a]), np.concatenate(p2 + [b])
        if len(p1): # People left over without a partner at their level make the share at the same level fall short, so more look within it, smoothed over about 20 steps (as for departures: see end_times())
            for k,(share,level) in enumerate(zip(self.pars.match.values(), levels)):
                self.match_extra[k] = np.clip(self.match_extra[k] + 0.05*(share - np.mean(level[p1] == level[p2])), -1, 1)
        return people[p1], people[p2], people[left]

    def end_times(self, p1, start=False):
        """ How many steps each new partnership lasts, or at the start, what is left of each partnership under way """
        if self.mean_dur is not None: # Geometric, so partnerships end at a constant rate (and what is left of one under way is the same)
            u = self.rng_end.rvs(p1)
            p_end = max(1/self.mean_dur - self.departures, 0.1/self.mean_dur) # Partnerships also end when a partner leaves, so they are dissolved less often, to last as long as given on average (as EpiModel's d.rate correction does)
            if p_end >= 1: # A mean of a step or less: each lasts one step (and log1p(-1) would divide by zero)
                return np.ones_like(u)
            return np.maximum(np.ceil(np.log1p(-u)/np.log1p(-p_end)), 1)
        n = len(p1)
        if start: # One picked in proportion to how long it lasts, and a uniform share of it
            rng = self.rng_stubs.rng
            dur = self.pars.dur.rvs(3*n)
            return dur[rng.choice(len(dur), size=n, p=dur/dur.sum())]*rng.random(n) if n else dur[:0]
        return self.pars.dur.rvs(p1 if len(np.unique(p1)) == n else n) # Drawn for each person, unless someone forms two at once (they would get the same)

    def pair(self, p1, p2, start=False):
        """ Add partnerships between two lists of people, leaving out pairs who are already partners (with a target; without, only people with none look), with their durations and acts """
        if len(self) and len(p1) and self.targeted:
            existing = set(zip(np.minimum(self.edges.p1, self.edges.p2).tolist(), np.maximum(self.edges.p1, self.edges.p2).tolist()))
            new = zip(np.minimum(p1, p2).tolist(), np.maximum(p1, p2).tolist())
            keep = np.array([pair not in existing for pair in new], dtype=bool)
            p1, p2 = p1[keep], p2[keep]
        keep = p1 != p2
        p1, p2 = p1[keep], p2[keep]
        if not len(p1):
            return 0
        acts = self.pars.acts
        if isinstance(acts, ss.Dist): # The acts in a step of each, from its acts per unit time: drawn once if a distribution, as for durations
            acts = dict(acts=acts.rvs(p1 if len(np.unique(p1)) == len(p1) else len(p1)).astype(ss_float))
        else:
            acts = {} if acts is None else dict(acts=np.full(len(p1), self.acts_per_step, dtype=ss_float))
        self.append(p1=p1, p2=p2, weight=np.ones(len(p1), dtype=ss_float), dur=self.end_times(p1, start=start), **acts)
        return len(p1)

    def init_post(self, add_pairs=True):
        """ Start at the target: give each person a number of partners, and pair them at random (with match=, some within their level)

        Args:
            add_pairs (bool): whether to add partnerships
        """
        if self.forms_own: # Start no partnerships for a subclass that forms its own, unless with its own add_pairs()
            super().init_post(add_pairs=add_pairs and type(self).add_pairs is not DegreeNet.add_pairs)
            return
        self.set_network_states()
        if not self.targeted or not add_pairs:
            super().init_post(add_pairs=add_pairs) # Pair everyone who takes part, as on each step (unless a subclass pairs its own)
            return
        super().init_post(add_pairs=False)
        uids = self.active(self.sim.people).uids
        rows = self.rows(uids)
        self.dist_degree = np.bincount(rows, minlength=len(self.dists_degree))/max(len(uids), 1) @ self.dists_degree # The population's target, from the mix of rows, for the results and check()
        cdfs = np.cumsum(self.dists_degree, axis=1)
        degrees = np.minimum((cdfs[rows] < self.rng_degree.rvs(uids)[:, None]).sum(axis=1), cdfs.shape[1] - 1)
        stubs = np.repeat(uids, degrees)
        fisher_yates_shuffle(stubs, self.rng_stubs.rng.random(len(stubs)))
        self.rng_stubs.jump() # Reset the RNG manually, as in RandomNet.shuffle_edges()
        rvs = None
        if self.pars.match is not None: # By position, since each person appears once for each partner
            rvs = self.rng_stubs.rng.random((len(self.pars.match), len(stubs)))
            self.rng_stubs.jump()
        p1, p2, left = self.match_pairs(stubs.view(ss.uids), rvs)
        self.pair(p1, p2, start=True)
        self.rng_stubs.jump()
        return

    def remove_uids(self, uids):
        """ Remove the partnerships of people who leave, counting them (see `end_times()`)

        Args:
            uids (array): the agents who leave
        """
        n = len(self)
        super().remove_uids(uids)
        self.lost += n - len(self)
        return

    def set_network_states(self):
        """ The states of people for the network, set on each step before pairing: none here (see `ss.SexualNet`) """
        pass

    def add_pairs(self):
        """ Pair the people looking for a partner: those who look on this step, given how many partners they have and the target, or with no target, everyone who has none """
        if len(self) or self.lost: # The share of partnerships ended by a partner leaving, smoothed over about 20 steps
            self.departures += 0.05*(self.lost/max(len(self) + self.lost, 1) - self.departures)
            self.lost = 0
        uids = self.active(self.sim.people).uids
        if self.targeted:
            prob = self.seek_prob(uids)
            seek = self.rng_seek.rvs(uids) < prob
            if len(self.waiting): # Those left over last step look again, unless they have as many partners as anyone can now
                seek |= np.isin(uids, self.waiting) & (prob > 0)
        else:
            seek = self.degrees(uids) == 0
        seekers = uids[seek]
        seekers = seekers[np.argsort(self.rng_order.rvs(seekers), kind='stable')]
        p1, p2, self.waiting = self.match_pairs(seekers) # Otherwise the surplus of one group would be lost on every step
        self.formed = self.pair(p1, p2)
        if self.targeted and self.mean_dur is None and len(uids):
            self.adjust(uids)
        return self.formed

    def adjust(self, uids):
        """
        With durations not at a constant rate, in an open population partners leave and newcomers start with none, so people look
        more (or less) often, by as much as keeps the mean number of partners at the target (a factor adjusted by a fifth of the
        gap each step, on a log scale), in each row of the targets
        """
        target = self.dists_degree @ np.arange(self.dists_degree.shape[1])
        degree = self.degrees(uids)
        rows = self.rows(uids)
        mean = np.array([degree[rows == r].mean() if np.any(rows == r) else t for r,t in enumerate(target)])
        gap = np.divide(target, np.maximum(mean, 1e-3*target), out=np.ones(len(target)), where=target > 0)
        self.boost = np.clip(self.boost*gap**0.5, 0.01, 1000.0)
        self.init_seek(self.mean_steps/(self.boost*gap)) # The gap itself too, which damps the swings of the slow adjustment
        return

    def step(self):
        """ End partnerships, set the states of people (e.g. debut), and form new partnerships """
        self.end_pairs()
        self.set_network_states()
        self.add_pairs()
        return

    def record_flows(self):
        """ The partnerships formed in the step, and the network of the step (made in its step, so recorded after it) """
        super().record_flows()
        if self.targeted:
            self.results.formed[self.ti] = getattr(self, 'formed', 0)
            degree = self.degrees(self.active(self.sim.people).uids)
            self.results.mean_degree[self.ti] = degree.mean() if len(degree) else 0
            self.results.concurrency[self.ti] = (degree >= 2).mean() if len(degree) else 0
        if len(self):
            for key,name in getattr(self, 'match_results', {}).items():
                level = self.levels(key)
                self.results[name][self.ti] = np.mean(level[self.edges.p1] == level[self.edges.p2])
        return

    def check(self, n_runs=3, output=False):
        """
        Run the sim again, and compare the network it makes with the network's targets

        Reports, for each target (the mean number of partners, the share with two or
        more, the maximum, and with `match=`, the share of partnerships at the same
        level), the target, the mean over the second half of each run, and the spread
        across runs; a target that is missed by more than the spread is a warning.
        With a number of partners for each level (`n_contacts=ss.by(...)`), each level has
        a row of its own (e.g. `mean_degree[high]`), and these and the rows for everyone are
        at the end of each run, with the targets for everyone those of the mix of levels then
        (which changes if, say, more die at one level).

        Args:
            n_runs (int): the number of runs
            output (bool): only return the table, without printing it

        Returns:
            The table, as a dataframe
        """
        rows = []
        mixes = [] # With a number of partners for each level, the share of people at each level at the end of each run
        for i in range(n_runs):
            run = self.sim.new(rand_seed=i + 1, verbose=0).run()
            net = run.networks[self.name]
            res = net.results
            half = len(res.mean_degree)//2
            row = dict(mean_degree=res.mean_degree[half:].mean(), concurrency=res.concurrency[half:].mean(), max_contacts=net.degrees(net.active(run.people).uids).max())
            if 'formed' in res and getattr(net, 'mean_dur', None) is not None and res.formed[half:].mean() > 0: # Little's law: partnerships = rate formed × how long they last
                n_alive = len(run.people.alive.uids)
                row['dur'] = res.mean_degree[half:].mean()*n_alive/2/res.formed[half:].mean()
            if getattr(net, 'by_level', None) is not None:
                uids = net.active(run.people).uids
                level = np.asarray(net.levels(net.by_level))[uids].astype(int)
                degree = net.degrees(uids)
                names = self.level_names(net.by_level)
                for lev in range(len(net.dists_degree)):
                    row[f'mean_degree[{names[lev]}]'] = degree[level == lev].mean() if np.any(level == lev) else np.nan
                mixes.append(np.bincount(level, minlength=len(net.dists_degree))[:len(net.dists_degree)]/max(len(level), 1))
                row.update(mean_degree=degree.mean() if len(degree) else 0, concurrency=(degree >= 2).mean() if len(degree) else 0) # At the end, as for each level, so for the same mix
            for key,name in getattr(self, 'match_results', {}).items():
                row[name] = res[name][half:].mean()
            rows.append(row)
        df = sc.dataframe(rows)
        dist = self.dist_degree if not mixes else np.mean(mixes, axis=0) @ np.array(self.dists_degree) # For everyone: with levels, for the mix of levels at the end of the runs
        k = np.arange(len(dist))
        targets = dict(mean_degree=dist @ k, concurrency=dist[2:].sum(), max_contacts=self.pars.get('max_contacts'), dur=getattr(self, 'mean_dur', None))
        if getattr(self, 'by_level', None) is not None:
            names = self.level_names(self.by_level)
            targets.update({f'mean_degree[{names[lev]}]':d @ np.arange(len(d)) for lev,d in enumerate(self.dists_degree)})
        for key,name in getattr(self, 'match_results', {}).items():
            targets[name] = self.pars.match[key]
        table = sc.dataframe(dict(target=[targets[c] for c in df.columns], realized=df.mean().values, spread=df.std().values), index=df.columns)
        for name,row in table.iterrows():
            if row.target is None:
                continue
            missed = row.realized > row.target if name == 'max_contacts' else abs(row.realized - row.target) > max(2*row.spread, 0.02*abs(row.target))
            if missed:
                ss.warn(f'the network "{self.name}" has {name} {row.realized:.3g}, against a target of {row.target:.3g}')
        if not output:
            print(table)
        return table

    def level_names(self, name):
        """ The name of each level of a property of people, for the rows of `check()`: a stratification's labels (e.g. low, high), or else the levels' numbers """
        strats = {strat.name:strat for strat in self.sim.groups.strats}
        n = len(self.dists_degree) if getattr(self, 'dists_degree', None) is not None else 0
        return [str(label) for label in strats[name].labels] if name in strats else [str(lev) for lev in range(n)]


class SexualNet(DegreeNet):
    """
    Sexual partnerships that form and dissolve, between two groups of people: by default, men and women

    A `ss.DegreeNet` between men and women (`between='mf'`), in which people take part
    after their debut (`debut`), if they do (`participation`, drawn at their debut). By
    default, people who have no partner are paired on every step, so nobody has more than
    one at a time. With `n_contacts`, the number of partners people have at a time is a
    target instead (among people who take part, after their debut), as for `ss.DegreeNet`;
    with partnerships much longer than people's time in the network, it takes a few
    decades to settle.

    Each partnership is between one person of each of two groups (`between`): a man and a
    woman, two men, or e.g. a female sex worker and a client, and each group can have its own
    number of partners (`n_contacts` as a dict). With `match=`, a share of partnerships is
    between people at the same level of a property of people, such as a risk group.

    Each partnership has a number of acts per unit time (`acts`), so a disease's beta on it is
    a probability per act (see `ss.peract`), and the share of them in which a condom is used
    (`condoms`, 0 unless set, e.g. by a subclass in `add_pairs()`), which a modifier of the
    transmission on the network reads, e.g.
    `ss.modify(hiv.infection, efficacy=0.8, share=lambda net: net.edges.condoms, network='sexualnet')`.

    A subclass that overrides `step()` forms its own partnerships: the base starts none for
    it, and has none of the parameters for forming them, only `between`.

    For several types of partnership with their own rules, see STIsim; for partnerships that
    are not sexual (e.g. needle sharing), `ss.DegreeNet`.

    Note: not CRN safe when people are born or die.

    Args:
        between (str/tuple): who partnerships are between: `'mf'` (the default), a man and a woman, so its directions are `m2f` and `f2m` and a disease's beta on it is `ss.peract(m2f=..., f2m=...)`, or one value for both; `'mm'`, two men (men who have sex with men), with one beta; or a pair of groups, each the name of a boolean property of people or of a level of a stratification, e.g. `('fsw', 'client')`, whose directions are `fsw2client` and `client2fsw` (nobody can be in both)
        dur (`ss.Dist`/`ss.Duration`/float): how long partnerships last, as for `ss.DegreeNet`: a single duration, e.g. `ss.months(6)`, is a mean, and partnerships end at a constant rate (an exponential); a bare number is in the sim's unit; give a distribution for any other spread, e.g. `ss.lognormal(mean=ss.months(6), std=ss.months(3))`, which can vary by age, year, and individual pair (default: `ss.lognormal(mean=ss.years(15), std=ss.years(1))`)
        debut (`ss.Dist`): the age at which people start taking part (default: normal, mean 16); its parameters can be functions, e.g. to vary it by sex or over time
        acts (`ss.freq`/`ss.Dist`): the number of acts per unit time in each partnership: a frequency (e.g. `ss.freqperweek(2)`, or `ss.freqperyear(80)`; a rate, `ss.peryear(80)`, is the same), or a distribution of them drawn once for each partnership (default: lognormal, mean 80 a year, standard deviation 30)
        participation (`ss.Dist`): whether each person takes part, drawn at their debut (default: 90% do); its parameters can be functions, e.g. to vary it by age or sex
        n_contacts (float/`ss.Dist`/dict): if given, a target for the number of partners people have at a time: a number, which is the mean of a Poisson, or a distribution over whole numbers; or one for each group of `between`, e.g. `dict(fsw=3, client=0.5)` (each partnership has one of each, so these should give the two groups about as many partners in all)
        concurrency (float): with `n_contacts`, the share of people with two or more partners at once
        max_contacts (int): with `n_contacts`, the most partners anyone has at once
        match (dict): if given, the share of partnerships between people at the same level of a property of people (a stratification, or an attribute), e.g. `dict(risk=0.8)`
        kwargs (dict): the other parameters of `ss.DegreeNet` (e.g. `who`, `memory`)
        pars (dict): the same parameters, as a dict

    Examples:
        ```python
        net = ss.SexualNet() # Men and women, at most one partner at a time
        msm = ss.SexualNet(name='msm', between='mm', participation=0.1, dur=ss.lognormal(mean=ss.years(2), std=ss.years(1)))
        net = ss.SexualNet(n_contacts=ss.poisson(0.7), dur=ss.lognormal(mean=ss.years(2), std=ss.years(1)), concurrency=0.1, match=dict(risk=0.8))
        sw = ss.SexualNet(name='sw', between=('fsw', 'client'), n_contacts=dict(fsw=5, client=0.5), dur=ss.months(1)) # With boolean attributes fsw and client
        ```
    """
    def __init__(self, pars=None, between=_, dur=_, debut=_, acts=_, participation=_, n_contacts=_, concurrency=_, max_contacts=_, match=_, **kwargs):
        ss.v3_deprecations.check(type(self), '__init__', list(kwargs) + list(pars or {}))
        super().__init__()
        self.define_pars(between='mf')
        if not self.forms_own: # A subclass that forms its own partnerships uses none of these
            self.define_pars(
                n_contacts = None,
                dur = ss.lognormal(mean=ss.years(15), std=ss.years(1)),  # Can vary by age, year, and individual pair
                acts = ss.lognormal(ss.freqperyear(80), ss.freqperyear(30)), # Acts per unit time, drawn once for each partnership
                debut = ss.normal(mean=16),
                participation = ss.bernoulli(p=0.9), # Drawn at debut
            )
        given = dur if dur is not None else (pars or {}).get('dur')
        single = not self.forms_own and (isinstance(given, ss.Duration) or sc.isnumber(given))
        if single: # A single duration (or a bare number, in the sim's unit) is a mean, and partnerships end at a constant rate, as for ss.DegreeNet: it replaces the default's distribution rather than setting its mean
            pars = {k:v for k,v in (pars or {}).items() if k != 'dur'}
            dur = None # So update_pars() does not read it from this frame
        self.update_pars(pars, **kwargs)
        if single:
            self.pars['dur'] = given
        self.set_between()
        self.meta.acts = ss_float # Acts in each partnership in a step: its acts per unit time times the step, a real number
        self.meta.condoms = ss_float # The share of acts with a condom, e.g. for ss.modify(..., share=lambda net: net.edges.condoms, network=)
        self.edge_defaults['condoms'] = 0.0
        self.debut = ss.FloatArr('debut') # Drawn for each person on their first step in the sim, whether born, arrived, or there at the start
        self.decided = ss.BoolArr('decided') # Whether a person's participation has been drawn, which happens at their debut, so a probability by age is read at that age
        return

    def active(self, people):
        """ The agents who can have partners, as a boolean array: alive, taking part, past their debut, and in a group of `between`

        Args:
            people (People): the people
        """
        return self.participant & (people.age > self.debut) & super().active(people)

    def available(self, people, sex):
        """ The UIDs of the agents of the given sex who can have partners and have none now

        Args:
            people (People): the people
            sex (str): 'male' or 'female'
        """
        available = people[sex] & self.active(people)
        available[self.edges.p1] = False
        available[self.edges.p2] = False
        return available.uids

    def set_network_states(self):
        """ Draw the age of debut of anyone who has none (newborns, and people who arrive at any age), and whether each person takes part, when they reach their debut """
        people = self.sim.people
        debuts = lambda reached: ss.solvers.backends.passes.debuts(people.auids, people.alive.raw, self.decided.raw, people.age.raw, self.debut.raw, reached).view(ss.uids)
        uids = debuts(False) # Alive, with no debut age
        self.debut[uids] = self.pars.debut.rvs(uids)
        uids = debuts(True) # Alive, at least their debut age, and not decided
        self.participant[uids] = self.pars.participation.rvs(uids)
        self.decided[uids] = True
        return


def find_pregnancy(net):
    """ The sim's `ss.Pregnancy`, which makes the edges of a prenatal or postnatal network; raises an error if it does not have exactly one """
    pregs = [mod for mod in net.sim.modules if isinstance(mod, ss.Pregnancy)]
    if len(pregs) != 1:
        raise RuntimeError(f'{type(net).__name__} requires one ss.Pregnancy, which makes its edges; the sim has {len(pregs)}')
    return pregs[0]


class PrenatalNet(Network):
    """
    A network from each pregnant woman to her unborn child, for transmission in pregnancy

    Its edges change over time, but `ss.Pregnancy` adds and removes them as pregnancies start and
    end (with the unborn as agents), so it is not a `DynamicNetwork`: it does not update itself.
    It transmits one way only, from the mother (`p1`) to the child (`p2`), so its one direction is `mother2child`.

    Note: CRN safe.
    """
    sides = dict(mother='p1', child='p2') # Where they are in an edge
    one_way = True

    def init_pre(self, sim):
        """ Check that the sim has the ss.Pregnancy that makes the edges

        Args:
            sim (Sim): the sim being initialized
        """
        super().init_pre(sim)
        find_pregnancy(self)
        return

    def step(self):
        """ Nothing: `ss.Pregnancy` adds and removes the edges """
        pass

    def add_pairs(self, mother_uids=None, unborn_uids=None):
        """ Add an edge from each mother to her unborn child; called by `ss.Pregnancy`

        Args:
            mother_uids (array): the mothers
            unborn_uids (array): the unborn children
        """
        n = len(mother_uids) if mother_uids is not None else 0
        if n:
            self.append(p1=mother_uids, p2=unborn_uids, weight=np.ones(n))
        return n


class PostnatalNet(DynamicNetwork):
    """
    A network from each mother to her infant after birth, for transmission after birth (e.g. by close contact)

    `ss.Pregnancy` (which the sim needs) adds an edge from the mother to each infant at birth,
    and it lasts for `dur`, so several of these networks can have postnatal periods of their own,
    e.g. one for each route of transmission. A disease transmits on it if its beta on it is above 0.

    The duration is drawn by the mother's UID, so it is the same for all her infants
    (a subclass can change this).
    It transmits one way only, from the mother (`p1`) to the child (`p2`), so its one direction is `mother2child`.

    Note: CRN safe.

    Args:
        pars (dict): passed to `ss.DynamicNetwork`
        dur (ss.Duration/ss.Dist): edge duration (postpartum period); if None, edges persist until agents die
        **kwargs (dict): merged with `pars`
    """

    sides = dict(mother='p1', child='p2') # Where they are in an edge
    one_way = True

    def __init__(self, pars=None, dur=None, **kwargs):
        super().__init__()
        self.define_pars(dur=dur)
        self.update_pars(pars, **kwargs)
        return

    def init_pre(self, sim):
        """ Find the sim's ss.Pregnancy, which makes the edges

        Args:
            sim (Sim): the sim being initialized
        """
        super().init_pre(sim)
        self.pregnancy = find_pregnancy(self)
        return

    def add_pairs(self, mother_uids=None, infant_uids=None):
        """ Add an edge from each mother to her infant

        Args:
            mother_uids (array): the mothers, with repeats if a mother has several infants
            infant_uids (array): the infants
        """
        n = len(mother_uids) if mother_uids is not None else 0
        p = self.pars
        if n:
            if isinstance(p.dur, ss.Dist):
                dur = p.dur.rvs(mother_uids) + 1 # +1 since edges are added by Pregnancy before end_pairs() decrements them in the same step
            elif p.dur is None:
                dur = np.full(n, np.inf)
            else:
                dur = np.full(n, p.dur/self.t.dt + 1)
            self.append(p1=mother_uids, p2=infant_uids, weight=np.ones(n), dur=dur)
        return n


class BreastfeedingNet(PostnatalNet):
    """
    A network from each mother to her infant while she breastfeeds, for transmission by breastfeeding

    The sim needs an `ss.Pregnancy`, which adds an edge at birth; it ends when the mother stops
    breastfeeding (`preg.lactating`).

    Note: CRN safe.

    Args:
        pars (dict): passed to `ss.PostnatalNet`
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(pars=pars, **kwargs)
        if self.pars.dur is not None:
            errormsg = 'BreastfeedingNet does not accept a `dur` parameter as breastfeeding duration is determined by the Pregnancy module'
            raise ValueError(errormsg)
        return

    def end_pairs(self):
        """ Remove edges for mothers who have stopped breastfeeding (`preg.lactating`) """
        active = self.pregnancy.lactating[self.p1]
        for k in self.meta_keys():
            self.edges[k] = self.edges[k][active]
        return np.count_nonzero(active)


#%% Mixing pools

class MixingPools(Route):
    """
    Mixing between groups of people, by a matrix of contacts between them: a mixing pool (`ss.MixingPool`) for each pair of groups, without edges

    Usually made by `ss.contacts()`, from a contact matrix. With no edges, mixing pools suit mixing
    in which who meets whom matters only by group (e.g. age group), not person by person.

    Every communicable disease transmits over them; a disease's beta on them is the disease's, by route (0 for one that does not spread here).

    Note: CRN safe as agents.

    Args:
        source (dict): the infectious groups, by name: each a condition over people (e.g. `ss.person.age < 15`), UIDs, or a function of the sim
        target (dict): the susceptible groups; as above
        n_contacts (array/`ss.Dist`): `n_contacts[i,j]`, the number of contacts that one person in target group `i` has with people in source group `j` (rows are the people who have the contacts, as in a contact survey and `ss.contacts()`). A matrix of numbers is the same for everyone in each group. One distribution whose parameters are matrices (e.g. `ss.poisson(M)`, or `ss.nbinom(mean=M, n=2)` for more spread) is drawn for each person when they enter a target group, and kept while they are in it. A matrix of distributions (one for each entry) is also drawn for each person, but is slower (a mixing pool for each pair of groups)
        pars (dict): the same parameters, as a dict

    Examples:
        ```python
        import starsim as ss
        mps = ss.networks.MixingPools(
            source = {'0-15': ss.person.age < 15, '15+': ss.person.age >= 15},
            target = {'0-15': ss.person.age < 15, '15+': ss.person.age >= 15},
            n_contacts = [[2.4, 0.91], [0.49, 0.16]],  # Illustrative contact matrix: row i, the contacts of a person in target group i
        )
        sim = ss.Sim(diseases='sis', networks=mps).run()
        sim.plot()

        # Contacts that differ between people: a Poisson draw for each person, with the matrix as its mean
        mps = ss.networks.MixingPools(source=mps.pars.source, target=mps.pars.target, n_contacts=ss.poisson([[2.4, 0.91], [0.49, 0.16]]))
        ```

    Transmission to a susceptible person `s` is `1 − (1 − p)^(rel_sus_s · Σ_j c_sj · trans_j)`, for the disease's chance `p` over the step, `s`'s contacts `c_sj`
    with source group `j`, and that group's mean infectiousness `trans_j`. This is exact for a force of infection proportional to the contacts and the
    mean infectiousness. For discrete contacts with random members of a group, it is a mean-field approximation: it uses the group's mean
    infectiousness rather than that of the people met, which slightly overstates infection (by a relative `p(1 − f)/2` when a share `f` of the
    group is infectious). It is exact when everyone in the group is equally infectious, and close when `p` is small.
    """
    def __init__(self, pars=None, source=_, target=_, n_contacts=_, **kwargs):
        ss.v3_deprecations.check(type(self), '__init__', list(kwargs) + list(pars or {}))
        super().__init__()
        self.define_pars(
            source = None,
            target = None,
            n_contacts = None,
        )
        self.update_pars(pars, **kwargs)
        self.validate_pars()
        self.contact_dist = None # With one distribution for the contacts: a copy, drawn one source group at a time (see draw_contacts())
        self.contact_scale = 1.0 # What its draws are multiplied by, to be per unit of the sim's time (see to_unit())
        if isinstance(self.pars.n_contacts, ss.Dist) and not isinstance(self.pars.n_contacts, ss.constant):
            self.contact_dist = sc.dcp(self.pars.n_contacts)
            self.define_states(
                ss.IntArr('contact_group', default=-1, label='Target group the contacts were drawn for'),
                *[ss.FloatArr(f'contacts_{j}', default=0.0, label=f'Contacts with source group {j}') for j in range(len(self.pars.source))],
            )
        self.pools = [] # A pool for each pair of groups, only if contacts differ between people and are a matrix of distributions: see plain_contacts()
        self.matrix = None # Otherwise, the number of contacts between groups, [target, source]
        self.source_uids = None # With a matrix: the people in each source group and each destination group, from step()
        self.target_uids = None
        self.diseases = None
        self.edges = None # The lower edges of the matrix's own age bands (from ss.contacts()), whose exposure agents work out by band: see band_exposure()
        self.dims = None # The matrices over the model's own dimensions (from ss.contacts())
        self.age_edges = None # The lower edges of the age matrix's own groups, if given with ages= (see ss.solvers.Mixing.dim_matrices())
        self.full = None # One matrix over several of the model's dimensions together (from ss.contacts())
        self.p_acquire = ss.bernoulli(p=0) # Placeholder value, for a matrix with no pools: see compute_transmission()
        self.unit = None # The unit of time the contacts are counted over, if not the sim's (e.g. 'day' for ss.contacts('Kenya')): see to_unit()
        return

    def to_unit(self, sim):
        """ Put the contacts in the sim's unit, once: a matrix counted over a unit of its own (`ss.contacts('Kenya')`, contacts a day) is scaled to contacts per unit of the sim's time """
        if self.unit is not None:
            factor = ss.solvers.convert(getattr(ss, f'per{self.unit.removesuffix("s")}')(1.0), sim.unit_base)
            if isinstance(self.pars.n_contacts, ss.Dist): # Each person's contacts are drawn in the matrix's unit, then scaled: see draw_contacts()
                self.contact_scale = factor
            else:
                self.pars.n_contacts = self.pars.n_contacts*factor
            if self.dims is not None:
                self.dims = {dim:mat*factor for dim,mat in self.dims.items()}
            if self.full is not None:
                self.full.matrix = self.full.matrix*factor
            self.unit = None
        return

    def __len__(self):
        """ The number of pools """
        try:    return len(self.pools)
        except Exception: return 0 # Before the pools exist

    def validate_pars(self):
        """ Check that source and target are dicts of groups, and n_contacts has a row for each target group (who has the contacts) and a column for each source group """
        p = self.pars
        if not isinstance(p.source, dict):
            errormsg = f'source= of ss.networks.MixingPools is a dict of groups, not {type(p.source)}'
            raise TypeError(errormsg)
        if not isinstance(p.target, dict):
            errormsg = f'target= of ss.networks.MixingPools is a dict of groups, not {type(p.target)}'
            raise TypeError(errormsg)
        p.source = sc.obj(p.source)
        p.target = sc.obj(p.target)
        if p.n_contacts is None:
            p.n_contacts = np.ones((len(p.target), len(p.source)))
        expected = (len(p.target), len(p.source))
        if isinstance(p.n_contacts, ss.Dist): # One distribution: its parameters are numbers or matrices
            try:
                np.broadcast_shapes(expected, *[np.shape(v) for v in p.n_contacts.pars.values() if sc.isarray(v)])
            except ValueError as e:
                errormsg = f'The parameters of the distribution of contacts (ss.{type(p.n_contacts).__name__}()) are a number, or a matrix with a row for each target group (who has the contacts) and a column for each source group, so shape {expected}'
                raise ValueError(errormsg) from e
            return
        p.n_contacts = np.array(p.n_contacts)
        actual = p.n_contacts.shape
        if actual != expected:
            errormsg = f'The mixing matrix has a row for each target group (who has the contacts) and a column for each source group, so shape {expected}, not {actual}.'
            raise ValueError(errormsg)
        return

    def check_full(self, sim):
        """ For one matrix over several stratifications together (`ss.contacts(matrix, dims=[...])`): check its shape against the levels of each """
        full = self.full
        names = [strat.name for strat in sim.groups.strats]
        for dim in full.dims:
            if dim not in names:
                raise sc.KeyNotFoundError(f'the contacts "{self.name}" are given by "{dim}", but the model is not stratified by it. Add e.g. sim.stratify({dim}=[...]).')
        strats = [sim.groups.strats[names.index(dim)] for dim in full.dims]
        counts = tuple(strat.n for strat in strats)
        if (full.levels is not None and tuple(full.levels) != counts) or len(full.matrix) != np.prod(counts):
            levels = '; '.join(f'"{strat.name}" has {strat.n} ({sc.strjoin(strat.labels)})' for strat in strats)
            shape = full.matrix.shape if full.levels is None else tuple(full.levels)*2
            raise ValueError(f'the contact matrix of "{self.name}" has shape {shape}, but {levels}, so it needs shape {counts*2} (or {(int(np.prod(counts)),)*2}, the first stratification slowest).')
        return

    def plain_contacts(self):
        """ The contacts as an array of numbers, if each is the same for everyone in its group; else None """
        p = self.pars
        if isinstance(p.n_contacts, ss.Dist):
            return np.broadcast_to(p.n_contacts.pars.value, (len(p.target), len(p.source))).astype(float) if isinstance(p.n_contacts, ss.constant) else None
        values = [n.pars.value if isinstance(n, ss.constant) else n for n in p.n_contacts.flat]
        if all(sc.isnumber(v) for v in values):
            return np.array(values, dtype=float).reshape(p.n_contacts.shape)
        return None

    def init_pre(self, sim):
        """ Create a mixing pool for each pair of groups, unless the contacts between groups can be used directly

        Args:
            sim (Sim): the sim being initialized
        """
        super().init_pre(sim)
        self.to_unit(sim)
        p = self.pars
        time_args = {k:p.get(k) for k in ss.Timeline.time_args} # get() allows None
        self.pools = []
        if self.edges is not None or self.dims is not None or self.full is not None or self.name in sim.mixing.pool_terms: # Used from the counts in each group or band, not with a pool for each pair: see ss.solvers.Mixing.init_pools()
            return
        self.matrix = self.plain_contacts()
        if self.matrix is not None or self.contact_dist is not None: # Everyone in a group has the same contacts, or one distribution: one pass over the groups does the work of a pool for each pair
            return
        for i,dk,target in p.target.enumitems():
            for j,sk,source in p.source.enumitems():
                n_contacts = p.n_contacts[i,j] # A number is the same for everyone in the group, as with no pools (a distribution elsewhere in the matrix does not change that)
                name = f'{self.name}:{sk}->{dk}' # Named after the whole, so that two of them (e.g. home and school contacts) can be in one sim
                mp = MixingPool(name=name, n_contacts=n_contacts, source=source, target=target, **time_args)
                mp.init_pre(sim)
                self.pools.append(mp)
        return

    def agent_reasons(self, sim):
        """ A number of contacts that differs between people is approximated by its mean in a compartmental run """
        nc = self.pars.n_contacts
        dists = [nc] if isinstance(nc, ss.Dist) else [n for n in nc.flat if isinstance(n, ss.Dist)]
        if any(not isinstance(d, ss.constant) for d in dists):
            return [ss.modules.Preference(f'the mixing pools "{self.name}" have numbers of contacts that differ between people')]
        return []

    def mean_matrix(self, unit):
        """ The mean number of contacts, [target, source], per unit of the sim's time (`unit`), as a compartmental run uses them """
        nc = self.pars.n_contacts
        if isinstance(nc, ss.Dist): # One distribution of matrices, as a matrix of distributions, one for each entry
            shape = (len(self.pars.target), len(self.pars.source))
            pars = {k:np.broadcast_to(v, shape) if sc.isarray(v) else v for k,v in nc.pars.items()}
            options = dict(unit=nc.unit) if nc.unit is not None else {}
            nc = [[type(nc)(**{k:v[i,j] if sc.isarray(v) else v for k,v in pars.items()}, **options) for j in range(shape[1])] for i in range(shape[0])]
        return self.contact_scale*np.vectorize(lambda n: ss.solvers.Mixing.mean(n, unit), otypes=[float])(np.array(nc, dtype=object))

    def init_post(self):
        """ Initialize each mixing pool, and find the diseases that use them """
        super().init_post()
        for mp in self.pools:
            mp.init_post()
        self.diseases = pool_diseases(self)
        return

    def band_exposure(self, weights):
        """
        As agents, for a matrix with its own age bands (`ss.contacts('Kenya')`), with no pool for each pair of bands: the
        UIDs of the people alive, and the exposure of each, from the infectiousness-weighted share of each band (`weights`,
        one for each agent) and the contacts between bands, as those pools would give, since everyone in a band has the
        same contacts. People younger than the first band (e.g. not yet born) have none
        """
        ppl = self.sim.people
        auids = ppl.auids
        band = ppl.age_bins(self.edges)[auids] # 0 if younger than the first
        nb = len(self.edges) + 1
        n = np.bincount(band, minlength=nb)
        infectious = np.bincount(band, weights=weights[auids], minlength=nb)
        n[0] = infectious[0] = 0
        prev = np.divide(infectious[1:], n[1:], out=np.zeros(nb-1), where=n[1:] > 0)
        rates = np.append(0.0, self.pars.n_contacts @ prev) # n_contacts is indexed [target, source]
        return auids, rates[band]

    def group_exposure(self, weights):
        """
        As agents, for a matrix of numbers: the UIDs of the people alive, and the exposure of each, as the pools for each pair of
        groups would give, from the infectiousness-weighted share of each source group (`weights`, one for each agent)
        """
        share = np.array([weights[uids].mean() if len(uids) else 0.0 for uids in self.source_uids])
        auids = self.sim.people.auids
        if self.contact_dist is not None: # Each person's own contacts with each source group
            expo = sum(self.state_dict[f'contacts_{j}'].raw[auids]*s for j,s in enumerate(share))
            return auids, expo
        expo = np.zeros(len(weights))
        for uids,value in zip(self.target_uids, self.matrix @ share): # A person in several target groups has the exposure of each
            expo[uids] += value
        return auids, expo[auids]

    def exposure(self, weights):
        """ As agents, with no pool for each pair of groups: see `band_exposure()` and `group_exposure()` """
        return self.band_exposure(weights) if self.edges is not None else self.group_exposure(weights)

    def compute_transmission(self, rel_sus, rel_trans, disease_beta, disease):
        """
        Calculate transmission, for a disease written as code: through each pool, or for a matrix, from the share of each group or band
        that is infectious, as a declared disease's is (see `ss.solvers.Agents.transmit()`), since everyone in a group has the same contacts

        Args:
            rel_sus (Arr): Relative susceptibility
            rel_trans (Arr): Relative infectiousness
            disease_beta (float): The beta value for the disease, as the probability in the step
            disease (Disease): The disease
        """
        if disease_beta == 0 or disease not in self.diseases:
            return []
        if self.pools:
            return [uid for mp in self.pools for uid in mp.compute_transmission(rel_sus, rel_trans, disease_beta, disease)]
        sim = self.sim
        if self.name in sim.mixing.pool_terms:
            auids = sim.people.auids
            groups = sim.groups.group_raw()[auids]
            n, infectious = [np.bincount(groups, weights=w, minlength=sim.groups.n_groups) for w in [None, rel_trans.raw[auids]]]
            expo = sim.mixing.term_exposure(sim.mixing.pool_terms[self.name], infectious, n.astype(float))[groups]
        else:
            auids, expo = self.exposure(rel_trans.raw)
        self.p_acquire.set(p=1 - (1 - disease_beta)**(expo*rel_sus.raw[auids]))
        return self.p_acquire.filter(auids)

    def remove_uids(self, uids):
        """ Remove UIDs from each mixing pool

        Args:
            uids (array): the agents to remove
        """
        for mp in self.pools:
            mp.remove_uids(uids)
        for key in ['source', 'target']:
            groups = self.pars[key]
            for k,group in groups.items():
                if isinstance(group, ss.uids):
                    groups[k] = group.remove(uids)
        return

    def step(self):
        """ Find the people in each group, once per step, and give them to each mixing pool (rather than each pool finding its own) """
        if not (self.pools or self.matrix is not None or self.contact_dist is not None): # From the counts in each group or band (ss.contacts()): see init_pre()
            return
        self.source_uids = [pool_uids(self.sim, group) for group in self.pars.source.values()]
        known = {id(group):uids for group,uids in zip(self.pars.source.values(), self.source_uids)} # A target group that is a source group (as for a matrix by age) is not found again
        self.target_uids = [known[id(group)] if id(group) in known else pool_uids(self.sim, group) for group in self.pars.target.values()]
        for k,mp in enumerate(self.pools): # Made target by target, then source by source: see init_pre()
            i, j = divmod(k, len(self.source_uids))
            mp.target_uids = self.target_uids[i]
            mp.source_uids = self.source_uids[j]
        if self.contact_dist is not None:
            self.draw_contacts()
        return

    def draw_contacts(self):
        """ With one distribution for the contacts: draw them for each person who has entered a target group since the last step, from its row of the matrix (none outside every target group) """
        group = np.full(len(self.contact_group.raw), -1)
        for i,uids in enumerate(self.target_uids):
            if (group[uids] >= 0).any():
                errormsg = f'Someone is in two target groups of "{self.name}", whose contacts are drawn from one distribution for each person, so the target groups cannot overlap'
                raise ValueError(errormsg)
            group[uids] = i
        auids = self.sim.people.auids
        moved = auids[group[auids] != self.contact_group.raw[auids]]
        if not len(moved):
            return
        self.contact_group[moved] = group[moved]
        into = moved[group[moved] >= 0]
        rows = group[into]
        shape = (len(self.pars.target), len(self.pars.source))
        pars = {k:np.broadcast_to(v, shape) for k,v in self.pars.n_contacts.pars.items() if sc.isarray(v)} # The parameters that differ by entry
        for j in range(shape[1]):
            contacts = self.state_dict[f'contacts_{j}']
            contacts[moved] = 0.0
            if len(into):
                self.contact_dist.set(**{k:v[rows, j] for k,v in pars.items()}) # A value for each person: their target group's entry
                draws = self.contact_dist.rvs(into)
                check_drawn(draws, self.name, self.pars.n_contacts)
                contacts[into] = draws*self.contact_scale
        return

    def shrink(self):
        """ Remove the people in each group, to save memory (e.g. before saving the sim) """
        super().shrink()
        for mp in self.pools:
            mp.shrink()
        shrunk = ss.utils.shrink()
        self.diseases = shrunk
        self.source_uids = shrunk
        self.target_uids = shrunk
        return


def age_band(group):
    """
    The ages a group of a mixing pool is for, as (low, high), if it is a condition on age alone: e.g. (5, 15) for
    `(ss.person.age >= 5) & (ss.person.age < 15)`, and (65, inf) for `ss.person.age >= 65`; else None. A compartmental run
    reads these to regroup the contacts to the model's age groups (see `ss.solvers.Mixing.age_matrix()`)

    Args:
        group (any): the group
    """
    Cond = ss.conditions.Cond
    if not isinstance(group, Cond):
        return None
    elif group.op == '&':
        a, b = age_band(group.args[0]), age_band(group.args[1])
        return None if a is None or b is None else (max(a[0], b[0]), min(a[1], b[1]))
    elif group.op in ['>=', '>', '<', '<=']:
        age, value = group.args
        if isinstance(age, Cond) and age.op == 'people' and age.args == ('age',) and sc.isnumber(value):
            return (value, np.inf) if group.op in ['>=', '>'] else (0, value)
    return None


def transmits(disease):
    """ Whether a disease is communicable: declared with a contact-mediated transition, or written as code with a beta """
    if disease.graph is not None:
        return len(disease.graph.transmissions) > 0
    return 'beta' in disease.pars


def pool_diseases(route):
    """ The diseases that transmit over a mixing pool or pools: every communicable disease (one whose beta on it is 0 transmits nothing there) """
    return [d for d in route.sim.diseases.values() if transmits(d)]


def pool_uids(sim, group):
    """ The people in a group of a mixing pool

    Args:
        sim (Sim): the sim
        group (Cond/func/uids): the people, as a condition over people (e.g. `ss.person.age < 15`), a function of the sim, or UIDs (default: all active agents)
    """
    if group is None:
        return sim.people.auids
    elif isinstance(group, (ss.conditions.Cond, str)):
        auids = sim.people.auids
        return auids[people_mask(sim, group)[auids]]
    elif callable(group):
        out = group(sim)
        return out.uids if isinstance(out, ss.BoolArr) else out # A boolean array (e.g. sim.people.female) is over the agents alive, so is converted to their UIDs
    elif isinstance(group, ss.uids):
        return group
    errormsg = f'A group of a mixing pool is a condition over people (e.g. ss.person.age < 5), a function of the sim, or UIDs, not {type(group)}'
    raise TypeError(errormsg)


class MixingPool(Route):
    """
    Mixing between two groups of people (or everyone), without edges: used in place of a network

    Each person in the target group is exposed to the average infectiousness of the source group,
    times their number of contacts. With no edges, it suits mixing in which who meets whom does
    not matter person by person. For mixing between several groups by a contact matrix, see
    `ss.contacts()`.

    Every communicable disease transmits over it; a disease's beta on it is the disease's, by route (0 for one that does not spread here).

    Note: CRN safe as agents.

    Args:
        source (Cond/uids/func): the infectious side: a condition over people (e.g. `ss.person.age < 15`), UIDs, or a function of the sim; None for everyone alive
        target (Cond/uids/func): the susceptible side; as above
        n_contacts (float/`ss.Dist`): the number of contacts of each target agent per unit of the sim's time (or a rate, e.g. `ss.perday(10)`): a number is the same for everyone, and a distribution is drawn for each agent (e.g. `ss.poisson(4)`)

    Examples:
        ```python
        import starsim as ss

        # Set the parameters
        mp_pars = dict(
            source = ss.person.male, # Only men are infectious
            target = None, # Everyone is susceptible
            n_contacts = ss.poisson(mean=4),
        )

        # Seed 5% of the male population
        def p_init(sim, uids):
            return 0.05*sim.people.male[uids]

        # Create and run the sim
        sis = ss.SIS(init_prev=p_init)
        mp = ss.MixingPool(**mp_pars)
        sim = ss.Sim(diseases=sis, networks=mp)
        sim.run()
        sim.plot()
        ```
    """
    def __init__(self, pars=None, source=_, target=_, n_contacts=_, **kwargs):
        ss.v3_deprecations.check(type(self), '__init__', list(kwargs) + list(pars or {}))
        super().__init__()
        self.define_pars(
            source = None,
            target = None, # Same as source
            n_contacts = ss.constant(10),
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.FloatArr('eff_contacts', default=self.pars.n_contacts, label='Effective number of contacts')
        )

        self.diseases = None
        self.source_uids = None
        self.target_uids = None
        self.p_acquire = ss.bernoulli(p=0) # Placeholder value
        return

    def agent_reasons(self, sim):
        """ A pool of some people has no compartmental form yet, and a number of contacts that differs between people is approximated """
        n_contacts = self.pars.n_contacts
        if self.pars.source is not None or self.pars.target is not None:
            return [f'the mixing pool "{self.name}" is between some people (source= or target=), which only agents have']
        elif isinstance(n_contacts, ss.Dist) and not isinstance(n_contacts, ss.constant):
            return [ss.modules.Preference(f'the mixing pool "{self.name}" has a number of contacts that differs between people ({type(n_contacts).__name__})')]
        return []

    def shrink(self):
        """ Remove the people in each group, to save memory (e.g. before saving the sim) """
        super().shrink()
        shrunk = ss.utils.shrink()
        self.diseases = shrunk
        self.source_uids = shrunk
        self.target_uids = shrunk
        self.setattribute('t', shrunk) # Its own copy of the timeline, for each of what can be many pools
        return

    def __len__(self):
        try:    return len(self.pars.target)
        except Exception: return 0 # E.g. a target of None, or before the parameters exist

    def init_pre(self, sim):
        """ A number of contacts given as a rate (e.g. `ss.perday(10)`) is a number per unit of the sim's time, as a bare number is

        Args:
            sim (Sim): the sim being initialized
        """
        n = self.pars.n_contacts
        rate = n.pars.value if isinstance(n, ss.constant) else n
        if isinstance(rate, ss.Rate):
            self.pars.n_contacts = ss.constant(ss.solvers.convert(rate, sim.unit_base))
            self.eff_contacts.default = self.pars.n_contacts
        super().init_pre(sim)
        return

    def init_post(self):
        """ Find the diseases that transmit over it, and check its contacts """
        super().init_post()
        self.diseases = pool_diseases(self)
        if isinstance(self.pars.n_contacts, ss.Dist):
            check_drawn(self.eff_contacts.values, self.name, self.pars.n_contacts) # Over the people at the start, not each new one, which would cost a check of each pool on every step
        return

    def remove_uids(self, uids):
        """ Remove people who leave (e.g. die) from source and target, if these are given as UIDs

        Args:
            uids (array): the agents to remove
        """
        for key in ['source', 'target']:
            inds = self.pars[key]
            if isinstance(inds, ss.uids):
                self.pars[key] = inds.remove(uids)
        return

    def compute_transmission(self, rel_sus, rel_trans, disease_beta, disease):
        """
        The people infected over the pool in a step

        This is called by transmission (`ss.Disease.transmit()`, declared, or as code from `ss.AgentDisease`) together with network transmission.

        Args:
            rel_sus (float): Relative susceptibility
            rel_trans (float): Relative infectiousness
            disease_beta (float): The beta value for the disease, as the probability in the step
            disease (Disease): The disease

        Returns:
            UIDs of agents who acquired the disease at this step
        """
        if disease_beta == 0 or disease not in self.diseases or len(self.source_uids) == 0 or len(self.target_uids) == 0:
            return []

        # Calculate transmission: the hazard of the step, from the disease's beta, is multiplied by the infectious share and the contacts before being turned back into a probability, as for a declared disease (see ss.solvers.Transmission)
        trans = np.mean(rel_trans[self.source_uids])
        acq = self.eff_contacts[self.target_uids] * rel_sus[self.target_uids]
        p = 1 - (1 - disease_beta)**(trans*acq) # I.e. 1 - exp(-hazard*dt), with the hazard -log(1 - disease_beta)/dt
        self.p_acquire.set(p=p)
        return self.p_acquire.filter(self.target_uids)

    def step(self):
        """ Update the UIDs of the source and target """
        self.source_uids = pool_uids(self.sim, self.pars.source)
        self.target_uids = pool_uids(self.sim, self.pars.target)
        return


def check_drawn(draws, name, dist):
    """ Raise an error for a negative number of contacts drawn from a distribution, which would be a negative chance of infection """
    if len(draws) and draws.min() < 0:
        errormsg = f'The contacts of "{name}" are drawn from ss.{type(dist).__name__}(), which gave {np.sum(draws < 0):,} negative numbers of contacts (e.g. {draws.min():n}); use a distribution of values that are not negative, e.g. ss.lognormal(), ss.gamma(), or ss.nbinom()'
        raise ValueError(errormsg)
    return


def check_contacts(matrix, dim):
    """ The entries of a contact matrix are numbers of contacts, so are finite and not negative """
    bad = ~np.isfinite(matrix) | (matrix < 0)
    if bad.any():
        i, j = np.argwhere(bad)[0]
        errormsg = f'The contact matrix of ss.contacts() for "{dim}" has {bad.sum()} entries that are negative or not finite, e.g. {matrix[i, j]} in row {i}, column {j}; each is a number of contacts per unit time'
        raise ValueError(errormsg)
    return


def contacts(matrix=None, ages=None, dims=None, name=None, study=None, layer='all', unit=None, **by_dim):
    """
    Mixing between groups from a contact matrix; returns `ss.networks.MixingPools`

    A matrix, and what it is over:

    - a country's, by name (`ss.contacts('Kenya')`), from `starsim_data`, by 5-year age group;
    - its own age groups, given by their lower edges (`ss.contacts(matrix, ages=[0, 15, 65])`), which need not be the model's (it is regrouped to them);
    - the model's own stratifications: its age groups (`ss.contacts(matrix)`), another one, by name (`ss.contacts(risk=matrix)`), or several together
      (`ss.contacts(matrix, dims=['sex', 'risk'])`), for mixing that is not a product, e.g. men mix with women, and by risk differently for each sex;
    - nobody in particular: a number (`ss.contacts(10)`) is homogeneous mixing, with that many contacts per person per unit time (an `ss.MixingPool` of everyone).

    Mixing that is a product over stratifications takes a matrix for each, by name (`ss.contacts(age=..., area=...)`): the contacts between
    two groups are the product of the entries for each. The matrix by age can have its own age groups here too (`ss.contacts(age=..., area=..., ages=[0, 15, 65])`).

    What the matrix multiplies is said by the rate of the infection, with I_j and N_j the number infectious and the number of people in group j, and N everyone:

    - `beta*I`: the share of each group who are infectious, β·Σ_j C_ij·I_j/N_j, with C the contacts of a person and β per contact (as for contact surveys);
    - `beta*I.n`: their number, β·Σ_j C_ij·I_j, density-dependent (as in Lloyd & Jansen's metapopulation);
    - `beta*I.frac` (or `beta*I.n/N`): their number over everyone, β·Σ_j C_ij·I_j/N (as in Keeling & Rohani's matrices, and WAIFW matrices).
    - a weighted count, `beta*(ss.by(w)*I).n` or `beta*I.n_infectious`: their number, each counted by its weight, β·Σ_j C_ij·w_j·I_j (for `n_infectious`, w_j is each group's infectiousness).

    So a textbook's matrix goes in as the textbook gives it, with no rescaling. See the user guide's page on population data for more.

    Args:
        matrix (array/str/df/float/`ss.Dist`): `matrix[i,j]`, the number of contacts per unit time that one person in group `i` has with people in group `j` (as in a contact survey); or a distribution whose parameters are matrices by age (with `ages=`) or tables by age group, for contacts that differ between people (e.g. `ss.contacts(ss.poisson(ss.data.contacts('Kenya')))`: see `ss.networks.MixingPools`); or a country, by name or code ('Kenya', 'KE'), or a US state ('United States:Massachusetts'); or a table by age group, as `ss.data.contacts()` gives (with labels such as '0-4' and '75+', or an index named "age"; a table labeled otherwise is a matrix as it is); or a number, for homogeneous mixing
        ages (list/array): the lower edge of each age group of the matrix (or of its matrix by age), e.g. `[0, 15, 65]` for 0–14, 15–64, and 65+
        dims (list): for one matrix over several of the model's stratifications together: their names, and `matrix[i, k, j, l]` is the number of contacts per unit time
            that one person at levels `(i, k)` has with people at levels `(j, l)`, or the same as a square matrix over the combinations of levels, the first stratification slowest
        name (str): the name of the mixing (default: 'contacts', or the layer, e.g. 'home')
        study (str): for a country, the study: 'prem' (the default: Prem 2021, bundled, for 177 countries, with Prem 2017 for 10 more, e.g. Japan and Australia), 'prem2021' (177), 'prem2017' (152), or 'mistry2021' (36, by single year of age); for a US state, 'mistry2021_us' (the default; by single year of age); all but Prem 2021 are downloaded on first use
        layer (str/list): for a country, the setting: 'all' (the default), 'home', 'school', 'work', or 'community'; or a list of them, for a list of layers, each named by its setting
        unit (str): the unit of time the contacts are counted over, e.g. 'day' for contacts a day; by default 'day' for a country's matrix (as the studies give them), and otherwise the sim's unit, as for any bare number
        by_dim (dict): a matrix over each of the model's stratifications, by name, e.g. `risk=[[2, 1], [4, 8]]`; with several, the contacts are the product of their entries

    Examples:
        ```python
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ 0.05*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01)
        mixing = ss.contacts([[10, 3], [3, 5]], ages=[0, 18])
        sim = ss.Sim(sir, networks=mixing, unit='day', stop=100).run()

        sir.stratify(risk=['low', 'high'])
        mixing = ss.contacts(risk=[[2, 1], [1, 8]])

        home, school = ss.contacts('Kenya', layer=['home', 'school']) # Contacts a day, by 5-year age group
        sim = ss.Sim(sir, networks=[home, school], stratify=dict(age=[0, 5, 18, 65]), unit='day', stop=100)

        # A product: by age, then 90% of contacts within one's own area
        mixing = ss.contacts(age=[[10, 3], [3, 5]], area=[[0.9, 0.1], [0.1, 0.9]], ages=[0, 18])

        # Keeling & Rohani's Program 3.1: a matrix of rates per year on shares of everyone, so the rate is I.frac
        S, I = ss.states('S I')
        sis = ss.Disease(S >> I @ I.frac, I >> S @ ss.peryear(1), init_prev=0.001).stratify(risk={'high': 0.2, 'low': 0.8})
        sim = ss.Sim(sis, networks=ss.contacts(risk=[[10, 0.1], [0.1, 1]]), unit='year', stop=15).run()

        # Men mix with women, and the high-risk with each other more among men than among women: a matrix over sex and risk together
        matrix = np.zeros((2, 2, 2, 2)) # [sex, risk, sex, risk], with sex ['female', 'male'] and risk ['low', 'high']
        matrix[0, :, 1, :] = [[0.8, 0.2], [0.5, 1.5]] # Women's contacts with men
        matrix[1, :, 0, :] = [[0.8, 0.5], [0.2, 2.0]] # Men's contacts with women
        mixing = ss.contacts(matrix, dims=['sex', 'risk'])

        # Contacts that differ between people: lognormal, around a country's mean contacts a day
        C = ss.data.contacts('Kenya')
        mixing = ss.contacts(ss.lognormal(mean=C, std=C*0.5))
        ```
    """
    if 'source' in by_dim: # Otherwise read as a matrix over a stratification called "source"
        errormsg = f"ss.contacts() takes no source= (the infectious side of a mixing pool): the study a country's matrix is from is study=, e.g. ss.contacts('Japan', study={by_dim['source']!r})"
        raise TypeError(errormsg)
    if isinstance(matrix, ss.Dist): # Contacts that differ between people, e.g. ss.contacts(ss.poisson(C), ages=[0, 15])
        return dist_contacts(matrix, ages=ages, name=name or 'contacts', unit=unit)
    if isinstance(layer, (list, tuple)): # Several settings, e.g. ss.contacts('Kenya', layer=['home', 'school', 'work', 'community'])
        return [contacts(matrix, ages=ages, dims=dims, study=study, layer=lay, unit=unit, **by_dim) for lay in layer]
    name = name or (layer if layer != 'all' else 'contacts')
    if isinstance(matrix, dict): # A matrix for each stratification is given by keyword
        errormsg = f'ss.contacts() takes a matrix for each stratification by name, e.g. ss.contacts({", ".join(f"{k}=..." for k in matrix)}), not as a dict'
        raise TypeError(errormsg)
    if isinstance(dims, str): # One stratification is a keyword
        errormsg = f'a matrix over one of the model\'s stratifications is given by its name: ss.contacts({dims}=matrix), not ss.contacts(matrix, dims={dims!r}); dims= is for one matrix over several together, e.g. dims=["sex", "risk"]'
        raise TypeError(errormsg)
    if by_dim:
        if matrix is not None or dims is not None:
            errormsg = f'ss.contacts() takes one matrix (with dims= for several stratifications together) or a matrix for each stratification by name ({sc.strjoin(by_dim)}), not both (the age groups of a matrix are ages=)'
            raise TypeError(errormsg)
        matrix = by_dim
    elif sc.isnumber(matrix): # Homogeneous mixing: a pool of everyone
        if ages is not None or dims is not None:
            errormsg = f'ss.contacts({matrix}) is homogeneous mixing, with {matrix} contacts per person per unit time, so it takes no ages= or dims='
            raise ValueError(errormsg)
        return MixingPool(n_contacts=getattr(ss, f'per{unit.removesuffix("s")}')(matrix) if unit is not None else matrix, name=name)
    if dims is not None and (isinstance(matrix, str) or ages is not None):
        errormsg = f'ss.contacts(matrix, dims={dims!r}) is a matrix over the model\'s own stratifications, so it takes no ages= (a matrix with its own age groups is ss.contacts(matrix, ages=[...])), and a matrix, not a name'
        raise ValueError(errormsg)
    ref = None
    if isinstance(matrix, str): # A country's matrix, from starsim_data, as a table by age group
        matrix = ss.data.contacts(matrix, study=study, layer=layer)
    if isinstance(matrix, pd.DataFrame): # A table by age group (e.g. ss.data.contacts('Kenya')), with the lower edge of each from its labels; any other table is a matrix over the model's own groups
        if ages is None and dims is None and ss.data.is_age_index(matrix.index):
            ages, ref = ss.data.age_edges(matrix.index), matrix.attrs.get('ref')
        matrix = matrix.values
    if dims is not None: # One matrix over several of the model's stratifications together
        over = sc.tolist(dims)
        matrix = np.array(matrix, dtype=float)
        k = len(over)
        if matrix.ndim == 2*k and matrix.shape[:k] == matrix.shape[k:]:
            n = int(np.prod(matrix.shape[:k]))
            levels = matrix.shape[:k]
            matrix = matrix.reshape(n, n)
        elif matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            levels = None # Known when the sim is built
        else:
            errormsg = f'ss.contacts(matrix, dims={over}) needs a matrix of shape (levels of each, then levels of each again), e.g. (2, 3, 2, 3) for 2 and 3 levels, or square over the combinations of levels, e.g. (6, 6); got shape {matrix.shape}'
            raise ValueError(errormsg)
        check_contacts(matrix, sc.strjoin(over, sep=' and '))
        out = MixingPools(source={}, target={}, n_contacts=np.zeros((0, 0)), name=name)
        out.full = sc.obj(dims=over, matrix=matrix, levels=levels) # So a compartmental run can use the matrix directly
        out.unit = unit
        return out
    if matrix is not None and not isinstance(matrix, (str, dict)) and ages is None: # Over the model's own age groups
        matrix = dict(age=matrix)
    age_edges = None
    if isinstance(matrix, dict) and ages is not None: # A matrix by age with its own age groups, with a matrix for each of other dimensions: regrouped to the model's age groups when the sim is built
        if 'age' not in matrix:
            errormsg = f'ss.contacts(..., ages=...) gives the age groups of the matrix by age, but there is none, only {sc.strjoin(matrix.keys())}; e.g. ss.contacts(age=[[...]], {list(matrix)[0]}=[[...]], ages=[0, 15, 65])'
            raise ValueError(errormsg)
        if len(matrix) == 1: # Only by age: a matrix with its own age groups
            matrix = matrix['age']
        else:
            mat = np.array(matrix['age'], dtype=float)
            if len(ages) != len(mat) or mat.ndim != 2 or mat.shape[0] != mat.shape[1] or np.any(np.diff(np.array(ages, dtype=float)) <= 0):
                errormsg = f'ss.contacts() needs a square matrix and the lower edge of each of its age groups, increasing, e.g. ss.contacts(matrix, ages=[0, 15, 65]); got a matrix of shape {mat.shape} and ages={list(ages)}'
                raise ValueError(errormsg)
            age_edges = np.asarray(ages) # As given (e.g. whole numbers)
            ages = None
    if isinstance(matrix, dict): # A matrix for each of the model's own dimensions
        mats = {dim:np.array(mat, dtype=float) for dim,mat in matrix.items()}
        for dim,mat in mats.items():
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                errormsg = f'The contact matrix of ss.contacts() for "{dim}" needs to be square, with a row and a column for each level of "{dim}"; got shape {mat.shape}'
                raise ValueError(errormsg)
            check_contacts(mat, dim)
        out = MixingPools(source={}, target={}, n_contacts=np.zeros((0, 0)), name=name)
        out.dims = mats # So a compartmental run can use the matrices directly
        out.age_edges = age_edges # The lower edges of the age matrix's own groups, if given with ages= (see ss.solvers.Mixing.dim_matrices())
        out.unit = unit
        return out
    if matrix is None:
        errormsg = 'ss.contacts() needs a matrix: a country (ss.contacts("Kenya")), a matrix with its own age groups (ss.contacts(matrix, ages=[...])), or one over the model\'s stratifications (ss.contacts(matrix), ss.contacts(risk=matrix))'
        raise ValueError(errormsg)
    matrix = np.array(matrix, dtype=float)
    if ages is None or len(ages) != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        errormsg = f'ss.contacts() needs a square matrix and the lower edge of each of its age groups, e.g. ss.contacts(matrix, ages=[0, 15, 65]); got a matrix of shape {matrix.shape} and ages={ages}'
        raise ValueError(errormsg)
    check_contacts(matrix, 'age')
    groups = age_groups(ages)
    out = MixingPools(source=groups, target=dict(groups), n_contacts=matrix, name=name) # ss.networks.MixingPools is indexed as a contact survey, [target, source]
    out.ref = ref # Where a named matrix came from
    out.edges = np.asarray(ages, dtype=float)
    out.unit = unit or ('day' if ref is not None else None) # A country's matrix is contacts a day, whatever the sim's unit
    return out


def age_groups(ages):
    """ The groups of a contact matrix by age, from the lower edge of each, e.g. {'0-15': ..., '15+': ...} for [0, 15] """
    if np.any(np.diff(np.array(ages, dtype=float)) <= 0):
        errormsg = f'The age groups of ss.contacts() are given by their lower edges, which need to increase, e.g. ages=[0, 15, 65]; got ages={list(ages)}'
        raise ValueError(errormsg)
    highs = list(ages[1:]) + [None]
    return {(f'{lo}-{hi}' if hi is not None else f'{lo}+'): (ss.person.age >= lo) & (ss.person.age < hi) if hi is not None else ss.person.age >= lo for lo,hi in zip(ages, highs)}


def dist_contacts(dist, ages, name, unit):
    """
    `ss.contacts()` for a distribution whose parameters are matrices by age, e.g. `ss.contacts(ss.poisson(C), ages=[0, 15])`, or tables by age
    group, as `ss.data.contacts()` gives them (whose age groups and unit, contacts a day, are then read from the table, as for the table itself).
    Each person draws their own contacts with each age group: see `ss.networks.MixingPools`
    """
    dist = sc.dcp(dist)
    tables = {k:v for k,v in dist.pars.items() if isinstance(v, pd.DataFrame)}
    ref = None
    if tables:
        first = next(iter(tables.values()))
        if any(not table.index.equals(first.index) for table in tables.values()):
            errormsg = f'The parameters of ss.{type(dist).__name__}() are tables by different age groups; give them by the same ones'
            raise ValueError(errormsg)
        if ages is None and ss.data.is_age_index(first.index):
            ages, ref = ss.data.age_edges(first.index), first.attrs.get('ref')
        dist.pars.update({k:table.values for k,table in tables.items()})
    if ages is None:
        errormsg = f'ss.contacts() with a distribution (ss.{type(dist).__name__}()) needs the lower edge of each of its age groups, e.g. ss.contacts(ss.poisson(C), ages=[0, 15, 65]), or its parameters as tables by age group, as ss.data.contacts() gives'
        raise ValueError(errormsg)
    groups = age_groups(ages)
    out = MixingPools(source=groups, target=dict(groups), n_contacts=dist, name=name)
    out.ref = ref
    out.unit = unit or ('day' if ref is not None else None) # A country's matrix is contacts a day, whatever the sim's unit
    return out
