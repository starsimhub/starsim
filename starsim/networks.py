"""
Networks that connect people within a population
"""
import itertools
import functools
import numpy as np
import numba as nb
import scipy.stats as sps
import scipy.sparse as spsp
import scipy.optimize as spo
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt
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
    intended behavior (see `ssl.RandomSafeNet` for the CRN-safe version).
    """
    n = arr.shape[0]
    for k in range(n - 1):
        i = n - 1 - k # Position to swap, working from the top down
        j = int(randvals[k] * (i + 1))
        tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp
    return arr


@nb.jit(cache=True) # No fastmath: a stray NaN must compare False (no transmission), not be assumed absent
def nb_transmit(src, trg, rel_trans, rel_sus, beta, randvals):
    """ Optimized transmission kernel: returns the (source, target) UIDs of transmitting edges.

    Fuses the gather/multiply/compare *and* the UID extraction into a single branchless pass.
    This avoids the full-length boolean temporary and the two separate gather passes
    (`trg[mask]`, `src[mask]`) of the previous approach -- ~1.4x faster at typical transmission
    rates, more when transmission is common. UIDs are emitted in edge order, preserving CRN behavior.
    """
    n = src.shape[0]
    src_out = np.empty(n, dtype=np.int64) # int64 = uid dtype, so the caller can .view(uids) without a copy
    trg_out = np.empty(n, dtype=np.int64)
    m = 0
    for i in range(n):
        transmitted = rel_trans[src[i]] * rel_sus[trg[i]] * beta[i] > randvals[i]
        src_out[m] = src[i] # Written every iteration (branchless); kept only if m advances
        trg_out[m] = trg[i]
        m += transmitted
    return src_out[:m], trg_out[:m]


# %% General network classes

def cond_str(cond):
    """ A condition over people, for messages: e.g. 'people.female' """
    if isinstance(cond, ss.conditions.Cond) and cond.op == 'people':
        return f'people.{cond.args[0]}'
    elif isinstance(cond, ss.conditions.Cond) and cond.op == '~' and isinstance(cond.args[0], ss.conditions.Cond) and cond.args[0].op == 'people':
        return f'not people.{cond.args[0].args[0]}'
    return getattr(cond, '__name__', str(cond))


def people_mask(sim, cond):
    """
    Who a condition over people holds for, as a boolean per agent (indexed by UID)

    Args:
        cond (Cond/str/func): a condition (`ss.ppl.pwid`), the name of a level of a stratification or of a boolean property of people (`'fsw'`), or a function of the sim
    """
    if isinstance(cond, str):
        strat = next((strat for strat in sim.groups.strats if cond in strat.labels), None)
        cond = ss.conditions.Cond('dim', strat.name) == cond if strat is not None else ss.conditions.Cond('people', cond)
    return np.asarray(ss.conditions.Selector(sim).select(cond), dtype=bool)


class Route(ss.Module):
    """
    A transmission route -- e.g., a network, mixing pool, environmental transmission, etc.
    """
    sets_unit = False # Its typed defaults (e.g. how long partnerships last) do not set the unit of the model's bare numbers: see ss.solvers.choose.init_unit()
    one_step = False # Whether every edge lasts one step: the edges of the dead then end at the next step anyway, so People.remove_dead() leaves them (see RandomNet)
    def compute_transmission(self, rel_sus, rel_trans, disease_beta, disease=None):
        """ Compute transmission probabilities along this route (must be overridden by subclasses). """
        errormsg = 'compute_transmission() must be defined by Route subclasses'
        raise NotImplementedError(errormsg)


class Network(Route):
    """
    A class holding a single network of contact edges (connections) between people
    as well as methods for updating these. Networks mediate disease transmission
    between agents in `ss.People`; see `ss.Disease` for how transmission uses
    network edges, and `ss.RandomNet`, `ss.SexualNet`, `ss.PrenatalNet` for
    built-in network types.

    The input is typically arrays including: person 1 of the connection, person 2 of
    the connection, the weight of the connection, the duration and start/end times of
    the connection.

    Note: CRN safe, with the edges given.

    Args:
        pars (dict): parameters, for subclasses that define them
        edges (dict/dataframe): the edges of the network, as an alternative to passing them as keyword arguments
        p1 (array): an array of length N, the number of connections in the network, with the indices of people on one side of the connection.
        p2 (array): an array of length N, the number of connections in the network, with the indices of people on the other side of the connection.
        beta (array): an array representing relative transmissibility of each connection for this network - TODO, do we need this?
        kwargs (dict): standard module arguments (e.g. `name`, `label`, `dt`) are passed to `update_pars()`; other keys are copied directly into the network as edges

    Note that all edge arguments must be arrays of the same length,
    although not all have to be supplied at the time of creation (they must all
    be the same at the time of initialization, though, or else validation will fail).

    Examples:
        ```python
        # Generate an average of 10 contacts for 1000 people
        n_contacts_pp = 10
        n_people = 1000
        n = n_contacts_pp * n_people
        p1 = np.random.randint(n_people, size=n)
        p2 = np.random.randint(n_people, size=n)
        beta = np.ones(n)
        network = ss.Network(p1=p1, p2=p2, beta=beta, label='rand')
        network = ss.Network(edges=dict(p1=p1, p2=p2, beta=beta), label='rand') # Alternate method

        # Convert one network to another with extra columns
        index = np.arange(n)
        self_conn = p1 == p2
        network2 = ss.Network(edges=network.edges, index=index, self_conn=self_conn, label=network.label)
        ```
    """

    def __init__(self, pars=None, edges=None, **kwargs):
        # Initialize as a module
        super().__init__()

        # Each relationship is characterized by these default set of keys
        self.meta = sc.objdict(
            p1 = ss_int,
            p2 = ss_int,
            beta = ss_float,
        )

        self.edge_defaults = {} # Values of edge columns that need not be given when edges are added (e.g. condoms=0 on a sexual network)

        # Initialize the keys of the network
        self.edges = sc.objdict()
        self.participant = ss.BoolArr('participant')

        # Standard module arguments are parameters; the rest are edges
        if 'sides' in kwargs: # The kinds of people at the two ends of its edges; see directions
            self.sides = kwargs.pop('sides')
        standard_args = ss.modules.module_args + ss.Timeline.time_args
        edges = dict(edges) if edges is not None else {} # dict() also handles dataframes
        edges.update({k:kwargs.pop(k) for k in list(kwargs) if k not in standard_args})
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
    def beta(self):
        """ Relative transmission on each network edge """
        return self.edges['beta'] if 'beta' in self.edges else None

    #%% Sides, directions, and acts

    sides = None # The kinds of people at the two ends of its edges: a dict of a short label to a condition over people (e.g. dict(m=~ss.ppl.female, f=ss.ppl.female)), from which its directions follow
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
        """ The direction of each edge from `src` to `trg`, as a position in `directions` (0 for a network with no sides); raises `ss.errors.RouteMismatch` for an edge with no direction """
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
            raise ss.errors.RouteMismatch(f'"{self.name}" has a partnership from {src[k]} to {trg[k]}, whose sides give no direction it has ({sc.strjoin(dirs)}; {self.name} has {self.sides_str()}){hint}')
        return out

    @property
    def custom_beta(self):
        """
        Whether it defines its own `net_beta()`: a declared disease then gives it the probability per act
        of each edge, in place of the hazard per act, and what it returns (the probability for the partnership in the step)
        is multiplied by `rel_trans`, `rel_sus`, and the modifiers on the route
        """
        return type(self).net_beta is not Network.net_beta

    def transmit(self, rel_trans, rel_sus, edge_beta, rng, prob=None):
        """
        The edges that transmit in a step, in each direction the network transmits in (p1 to p2, then p2 to p1 unless it is one way)

        An edge from `src` to `trg` transmits with probability `rel_trans[src]·rel_sus[trg]·beta`, or `prob()` if it is given.
        Random numbers are drawn only for the edges that can transmit: with CRN, each edge's number depends only on its source
        and target, so this gives the same result as drawing one for every edge. The beta is worked out only for the edges from
        someone infectious to someone susceptible, which are found for both directions in one pass.

        Args:
            rel_trans (array): the transmissibility of each agent, by UID (0 if not infectious)
            rel_sus (array): the susceptibility of each agent, by UID (0 if not susceptible)
            edge_beta (func): of the direction (0 or 1), and the indices, sources, and targets of the edges from someone with `rel_trans` not 0
                to someone with `rel_sus` not 0, the beta of each of them (an array, or a number for all; see `edge_net_beta()`): with `prob`, only
                where it is 0 matters (no transmission); None to skip the direction (no random numbers are drawn)
            rng (`ss.multi_random`): the random numbers, by source and target
            prob (func): of the direction, and the indices, sources, and targets of the edges that can transmit, the probability of each, in place of the product above

        Returns:
            A list of the (source UIDs, target UIDs) that transmit, for each direction not skipped
        """
        out = []
        if not len(self):
            return out
        edges = self.edges
        found = ss.perflib._passes.candidates(edges.p1, edges.p2, rel_trans, rel_sus, two_way=not self.one_way) # Mother-to-child networks transmit one way
        for d,inds in enumerate(found):
            src, trg = [ends[inds] for ends in self.ends(d)]
            beta = edge_beta(d, inds, src, trg)
            if beta is None:
                continue
            beta = np.asarray(beta, dtype=ss_float) if np.ndim(beta) == 0 else beta # One value for all edges
            can = ss.perflib._passes.can_transmit(src, trg, rel_trans, rel_sus, beta) # Of those, the ones with a chance above 0, exactly as over every edge
            if len(can) < len(inds):
                inds, src, trg = inds[can], src[can], trg[can]
                beta = beta[can] if np.ndim(beta) else beta
            rvs = rng.rvs(src, trg) # Called even if empty, to keep CRN draws in sync
            if prob is None:
                beta = np.broadcast_to(beta, len(inds)) if np.ndim(beta) == 0 else beta
                src, trg = nb_transmit(np.asarray(src), np.asarray(trg), rel_trans, rel_sus, beta, rvs)
            else:
                hit = rvs < prob(d, inds, src, trg)
                src, trg = np.asarray(src)[hit], np.asarray(trg)[hit]
            out.append((src.view(ss.uids), trg.view(ss.uids))) # View (no copy): the kernel's output is its own int64 array
        return out

    def init_results(self):
        """ Store network length by default """
        super().init_results()
        self.define_results(
            ss.Result('n_edges', dtype=int, scale=True, label='Number of edges', auto_plot=False)
        )
        return

    def init_pre(self, sim):
        """ Initialize with the sim, initialize the edges, and validate p1 and p2 """
        super().init_pre(sim)

        # Define states using placeholder values
        n = len(self.edges['p1']) if 'p1' in self.edges else 0
        for key, dtype in self.meta.items():
            if key not in self.edges:
                self.edges[key] = np.full(n, self.edge_defaults[key], dtype=dtype) if key in self.edge_defaults else np.empty((0,), dtype=dtype)

        # Check that p1 and p2 are ss.uids()
        self.validate_uids()
        return

    def init_post(self, add_pairs=True):
        super().init_post()
        if add_pairs: self.add_pairs()
        return

    def mean_contacts(self, sim):
        """
        For a compartmental run: the mean number of contacts per person per unit of the sim's time, which homogeneous mixing approximates
        this network by, and where it comes from; or None, and why, if it has no such approximation (the default). A network of
        partnerships has none: its contacts last, so their number per unit time is not its number of partners (it uses the
        partner-change model instead: see `ss.solvers.Mixing.partnerships()`).
        """
        return None, ''

    def __len__(self):
        """ The length is the number of edges """
        try:    return len(self.edges.p1)
        except: return 0

    def __repr__(self, **kwargs):
        """ Same as default, except with length """
        out = self.brief(output=True)
        len_str = f'n_edges={len(self)}; '
        pos = out.find('pars=')
        out = out[:pos] + len_str + out[pos:]
        return out

    def __str__(self):
        """ Slightly more detailed, show the dataframe as well """
        out = self.brief(output=True)
        out += '\n' + self.to_df().__repr__()
        return out

    def __contains__(self, item):
        """
        Check if a person is present in a network

        Args:
            item: Person index

        Returns: True if person index appears in any interactions
        """
        return (item in self.edges.p1) or (item in self.edges.p2) # TODO: check if (item in self.members) is faster

    @property
    def members(self):
        """ Return sorted array of all members """
        return np.unique([self.edges.p1, self.edges.p2]).view(ss.uids)

    def meta_keys(self):
        """ Return the keys for the network's meta information """
        return self.meta.keys()

    def set_network_states(self, people):
        """
        Many network states depend on properties of people -- e.g. MSM depends on being male,
        age of debut varies by sex and over time, and participation rates vary by age.
        Each time states are dynamically grown, this function should be called to set the network
        states that depend on other states.
        """
        pass

    def validate_uids(self):
        """ Ensure that p1, p2 are both UID arrays """
        edges = self.edges
        for key in ['p1', 'p2']:
            if key in edges:
                arr = edges[key]
                if not isinstance(arr, ss.uids):
                    self.edges[key] = ss.uids(arr)
        return

    def validate(self, force=True):
        """
        Check the integrity of the network: right types, right lengths.

        If dtype is incorrect, try to convert automatically; if length is incorrect,
        do not.
        """
        if not len(self.edges): # No edges have been added yet
            return
        n = len(self.edges.p1)
        for key, dtype in self.meta.items():
            if dtype:
                actual = self.edges[key].dtype
                expected = dtype
                if actual != expected:
                    self.edges[key] = np.array(self.edges[key], dtype=expected)  # Try to convert to correct type
            actual_n = len(self.edges[key])
            if n != actual_n:
                errormsg = f'Expecting length {n} for network key "{key}"; got {actual_n}'  # Report length mismatches
                raise TypeError(errormsg)
        self.validate_uids()
        return

    def eligible(self, age_range=None):
        """
        Return the agents who can form edges: alive, born, and (optionally) within an age range

        Args:
            age_range (list): if supplied, only include agents with `age_range[0] <= age < age_range[1]`
        """
        people = self.sim.people
        eligible = people.alive & (people.age > 0)
        if age_range is not None:
            eligible = eligible & (people.age >= age_range[0]) & (people.age < age_range[1])
        return eligible

    def get_inds(self, inds, remove=False):
        """
        Get the specified indices from the edgelist and return them as a dict.

        Args:
            inds (int, array, slice): the indices to find
            remove (bool): whether to remove the indices
        """
        output = {}
        for key in self.meta_keys():
            output[key] = self.edges[key][inds]  # Copy to the output object
            if remove:
                self.edges[key] = np.delete(self.edges[key], inds)  # Remove from the original
                self.validate_uids()
        return output

    def pop_inds(self, inds):
        """
        "Pop" the specified indices from the edgelist and return them as a dict.
        Returns arguments in the right format to be used with network.append().

        Args:
            inds (int, array, slice): the indices to be removed
        """
        popped_inds = self.get_inds(inds, remove=True)
        return popped_inds

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
        Append edges to the current network.

        Args:
            edges (dict): a dictionary of arrays with keys p1,p2,beta, as returned from network.pop_inds()
        """
        edges = sc.mergedicts(edges, kwargs)
        for key in self.meta_keys():
            curr_arr = self.edges[key]
            if key in edges:
                new_arr = edges[key]
            elif key in self.edge_defaults:
                new_arr = np.full(len(edges['p1']), self.edge_defaults[key], dtype=self.meta[key])
            else:
                errormsg = f'Cannot append edges since required key "{key}" is missing'
                raise KeyError(errormsg)
            self.edges[key] = np.concatenate([curr_arr, new_arr])  # Resize to make room, preserving dtype
        self.validate_uids()
        return

    def update_results(self):
        """ Store the number of edges in the network """
        super().update_results()
        self.results['n_edges'][self.ti] = len(self)
        return

    def to_graph(self, max_edges=None, random=False): # pragma: no cover
        """
        Convert to a networkx DiGraph

        Args:
            max_edges (int): the maximum number of edges to show
            random (bool): if true, select edges randomly; otherwise, show the first N

        Examples:
            ```python
            import networkx as nx
            sim = ss.Sim(n_agents=100, networks='sexual').init()
            G = sim.networks.randomnet.to_graph()
            nx.draw(G)
            ```
        """
        import networkx as nx # Lazy import since slow
        keys = [('p1', int), ('p2', int), ('beta', float)]
        data = [np.array(self.edges[k], dtype=dtype) for k,dtype in keys]
        if max_edges:
            if random:
                inds = np.sort(np.random.choice(len(data[0]), max_edges, replace=False))
                data = [col[inds] for col in data]
            else:
                data = [col[:max_edges] for col in data]
        G = nx.DiGraph()
        G.add_weighted_edges_from(zip(*data), weight='beta')
        nx.set_edge_attributes(G, self.label, name='layer')
        return G

    def to_edgelist(self):
        """ Convert the network to a list of edges (paired nodes) """
        out = list(zip(self.p1, self.p2))
        return out

    def to_dict(self):
        """ Convert to dictionary """
        d = {k: self.edges[k] for k in self.meta_keys()}
        return d

    def to_df(self):
        """ Convert to dataframe """
        df = sc.dataframe.from_dict(self.to_dict())
        return df

    def from_df(self, df, keys=None):
        """
        Set the edges from a dataframe or a dict of arrays

        Replaces the existing edges, and returns the network, so it can also be used to
        create a new network. If `beta` is not supplied, it defaults to 1.

        Args:
            df (dataframe/dict): the edges, with columns `p1`, `p2`, and (optionally) `beta`
            keys (list): the columns to use (default: the network's meta keys)

        Examples:
            ```python
            net = ss.Network(name='mynet').from_df(dict(p1=[0,1,2], p2=[1,2,3])) # Create a new network
            net.from_df(df) # Replace the edges of an existing network
            ```
        """
        if keys is None:
            keys = self.meta_keys()
        for key in keys:
            if key == 'beta' and key not in df:
                self.edges[key] = np.ones(len(df['p1']), dtype=ss_float)
            elif key in self.edge_defaults and key not in df:
                self.edges[key] = np.full(len(df['p1']), self.edge_defaults[key], dtype=self.meta.get(key))
            else:
                self.edges[key] = np.array(df[key], dtype=self.meta.get(key))
        self.validate_uids()
        return self

    def shrink(self):
        """ Shrink the size of the network for saving to disk """
        super().shrink()
        shrunk = ss.utils.shrink()
        self.edges = shrunk
        self.participant = shrunk
        return

    def plot(self, max_edges=500, random=False, alpha=0.2, **kwargs):
        """
        Plot the network using NetworkX.

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
                n_edges = len(self)
                if n_edges > max_edges:
                    edgestr = f'{max_edges:n} of {len(self):n} connections shown'
                else:
                    edgestr = f'{len(self):n} connections'
                plt.title(f'{self.label}: {edgestr}')
        return ss.return_fig(fig)

    def find_contacts(self, inds, as_array=True):
        """
        Find all contacts of the specified people

        For some purposes (e.g. contact tracing) it's necessary to find all the edges
        associated with a subset of the people in this network. Since edges are bidirectional
        it's necessary to check both p1 and p2 for the target indices. Each contact is only
        returned once (otherwise if the Network has explicit symmetric interactions, they could
        appear multiple times), so the calling code doesn't need to perform its own unique()
        operation. Note that this cannot be used for cases where multiple connections count
        differently than a single infection, e.g. exposure risk.

        Args:
            inds (array): indices of people whose edges to return
            as_array (bool): if true, return as sorted array (otherwise, return as unsorted set)

        Returns:
            contact_inds (array/set): the indices of the pairing partners

        Example: If there were a network with
        - p1 = [1,2,3,4]
        - p2 = [2,3,1,4]
        Then find_edges([1,3]) would return {1,2,3}
        """
        contact_inds = ss.utils.find_contacts(self.edges.p1, self.edges.p2, sc.toarray(inds), as_array=as_array)
        return contact_inds

    def add_pairs(self):
        """ Define how pairs of people are formed """
        pass

    def remove_uids(self, uids):
        """
        Remove interactions involving specified UIDs
        This method is typically called via `People.remove()` and
        is specifically used when removing agents from the simulation.
        """
        p1, p2 = self.edges.p1, self.edges.p2
        n = len(self.sim.people.alive.raw) if self.initialized else max(np.max(p1, initial=-1), np.max(p2, initial=-1)) + 1 # Every UID in the network
        is_removed = np.zeros(max(n, np.max(uids, initial=-1) + 1), dtype=bool) # A lookup table of the UIDs to remove (much faster than np.isin() for many edges)
        is_removed[uids] = True
        self.drop_edges(*ss.perflib._passes.touching(p1, p2, is_removed)) # Remove edges where either partner is being removed
        return

    def drop_edges(self, k, gone):
        """ Remove the `k` edges at the indices `gone` (in order; all edges if `k` is the number of edges), as `ss.perflib._passes.ending()` gives them; not for the user """
        if not k:
            return
        for key in self.meta_keys():
            arr = self.edges[key]
            self.edges[key] = arr[:0] if k == len(arr) else ss.perflib._passes.compress(arr, gone).view(type(arr))
        return

    def net_beta(self, disease_beta=None, inds=None, disease=None):
        """
        The probability that each edge transmits in a step, given the disease's probability (in the step, or on a network with acts, per act)

        A network written as code can override this (see `custom_beta`).
        """
        if inds is None: inds = Ellipsis
        if self.has_acts: # The number of acts in each partnership in the step
            return self.edges.beta[inds] * (1 - (1 - disease_beta) ** self.edges.acts[inds])
        return self.edges.beta[inds] * disease_beta # Beta should already include dt if desired

    def ends(self, d):
        """ The sources and targets of every edge, in direction `d`: p1 to p2 (0) or p2 to p1 (1) """
        return (self.edges.p1, self.edges.p2) if d == 0 else (self.edges.p2, self.edges.p1)

    def edge_net_beta(self, inds, disease_beta=None, disease=None):
        """ `net_beta()` for the edges at `inds`; one written as code (see `custom_beta`) is worked out for every edge, as in Starsim 3, so `disease_beta` is then a number or a value for every edge; not for the user """
        if self.custom_beta:
            beta = self.net_beta(disease_beta=disease_beta, disease=disease)
            return beta[inds] if np.ndim(beta) else beta
        return self.net_beta(disease_beta=disease_beta, inds=inds, disease=disease)


class DynamicNetwork(Network):
    """
    A network where partnerships update dynamically

    Args:
        memory (dur): how long it remembers partnerships that have ended (default: not at all); `find_contacts()`, and so
            contact tracing (`trace=` of a delivery), then also reaches the partners of the last `memory`, e.g. `memory=ss.months(12)`
            for partner notification of the past year's partners
    """
    def __init__(self, pars=None, edges=None, memory=None, **kwargs):
        super().__init__(pars=pars, edges=edges, **kwargs)
        self.define_pars(memory=memory)
        self.meta.dur = ss_float # Add duration to the meta keys for dynamic networks
        self.history = None # With memory=: the partnerships that ended, as p1, p2, and the step they ended on
        return

    def step(self):
        """ Remove expired partnerships and add new ones """
        self.end_pairs()
        self.add_pairs()
        return

    def end_pairs(self):
        """ Remove partnerships whose duration has expired or whose members have died. """
        n = len(self)
        k, gone = ss.perflib._passes.ending(self.edges.dur, self.edges.p1, self.edges.p2, self.sim.people.alive.raw) # Those with no steps left after this one, or a partner not alive
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
        h = self.history if self.history is not None else sc.objdict(p1=p1[:0], p2=p2[:0], ti=np.zeros(0, dtype=np.int32))
        keep = h.ti > ti - steps
        self.history = sc.objdict(p1=np.concatenate([h.p1[keep], p1]), p2=np.concatenate([h.p2[keep], p2]), ti=np.concatenate([h.ti[keep], np.full(len(p1), ti, dtype=np.int32)]))
        return

    def find_contacts(self, inds, as_array=True):
        """ Find all contacts of the specified people (see `ss.Network.find_contacts()`), including, with memory=, their partners of the last `memory` """
        if self.history is None:
            return super().find_contacts(inds, as_array=as_array)
        p1, p2 = np.concatenate([self.edges.p1, self.history.p1]), np.concatenate([self.edges.p2, self.history.p2])
        return ss.utils.find_contacts(p1, p2, sc.toarray(inds), as_array=as_array)


def check_acts(net, acts):
    """
    `acts=` of lasting partnerships is acts per unit time: a frequency (`ss.freq`; a hazard or probability is an error), or a distribution of them drawn once for each
    partnership; a bare number, or a count per step drawn once, would depend on the step, so is an error
    """
    name = net.name or type(net).__name__.lower()
    example = 'e.g. ss.freqperweek(2), or ss.lognorm_ex(ss.freqperyear(80), ss.freqperyear(30)) for a rate that varies between partnerships'
    if acts is None or isinstance(acts, ss.freq):
        return
    rates = [acts] if isinstance(acts, ss.Rate) else []
    if isinstance(acts, ss.Dist):
        rates = [v for v in acts.pars.values() if isinstance(v, ss.Rate) and not isinstance(v, ss.freq)]
    if rates: # E.g. ss.peryear(80), which as agents would be the chance of an act in a step, not a number of them
        raise ValueError(f'acts= of "{name}" is a number of acts per unit time, not a hazard or probability per unit time (ss.{rates[0]!r}): give a frequency, {example}')
    elif sc.isnumber(acts):
        raise ValueError(f'acts= of "{name}" is a number of acts per unit time in each partnership; a bare number ({acts}) would be acts in a step, which depends on the step. Give a rate, {example}')
    elif isinstance(acts, ss.Dist):
        timed = acts.unit is not None or any(isinstance(v, ss.TimePar) for v in acts.pars.values())
        if not timed:
            raise ValueError(f'acts= of "{name}" is a number of acts per unit time in each partnership, so a distribution of them needs its parameters as rates ({acts} has none), {example}')
        if isinstance(acts, (ss.poisson, ss.nbinom, ss.binom, ss.randint, ss.bernoulli)):
            raise ValueError(f'acts= of "{name}" is {acts}, a whole number of acts in a step, drawn once for each partnership, which depends on the step (at weekly steps, a fifth of partnerships would never have sex). Give a rate, {example}')
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
      agent has exactly `n_contacts` contacts.
    - `'uniform'`: uniformly from all eligible agents, regardless of their own
      `n_contacts`, as in Covasim v3's random networks. This gives less variable numbers
      of contacts (e.g. with `n_contacts=ss.poisson(20)`, a variance of about 15, rather
      than 30 for `'fast'` or 20 for `'exact'`), but loses the mixing structure above.

    Note: not CRN safe when people are born or die, unless `dynamic=False`: with any of these
    methods, one death rewires almost every edge. `ssl.RandomSafeNet` (`import starsim.library as ssl`)
    is a CRN-safe (but slower) version.

    Args:
        n_contacts (int/`ss.Dist`): the average number of (bidirectional) contacts between agents
        dur (int/`ss.dur`): the duration of each contact
        beta (float): the default beta value for each edge
        age_range (list): if supplied, only agents with `age_range[0] <= age < age_range[1]` are included
        dynamic (bool): if True (default), edges are replaced once their duration ends; if False, edges are created once and never change
            (so agents who join later, e.g. newborns, are not added)
        method (str): how the other end of each contact is drawn: 'fast' (the default), 'exact', or 'uniform' (see above)

    Note: n_contacts = 10 will create *5* edges per agent. Since disease transmission
    usually occurs bidirectionally, this means that the effective number of contacts
    per agent is actually 10. Consider 3 agents with 3 edges between them (a triangle):
    each agent is connected to 2 other agents.

    Example:
        ```python
        # Static network with Poisson-distributed contacts among adults, as in Covasim v3
        net = ss.RandomNet(n_contacts=ss.poisson(20), age_range=[18, 65], dynamic=False, method='uniform')
        ```
    """
    methods = ['fast', 'exact', 'uniform']

    def __init__(self, pars=None, n_contacts=_, dur=_, beta=_, age_range=_, dynamic=_, method=_, **kwargs):
        """ Initialize """
        super().__init__()
        self.define_pars(
            n_contacts = ss.constant(10),
            dur = ss.years(0), # Note; network edge durations are required to have the same unit as the network
            beta = 1.0,
            age_range = None,
            dynamic = True,
            method = 'fast',
        )
        self.update_pars(pars, **kwargs)
        if self.pars.method not in self.methods:
            errormsg = f'method= of ss.RandomNet() is one of {sc.strjoin(self.methods)}, not {self.pars.method!r}'
            raise ValueError(errormsg)
        self.dist = ss.Dist(distname='RandomNet') # Default RNG; the same name for every method, so results stay reproducible
        return

    def mean_contacts(self, sim):
        """ Its mean number of contacts (see `ss.Network.mean_contacts()`) """
        n_contacts = self.pars.get('n_contacts')
        if n_contacts is None:
            return super().mean_contacts(sim)
        return ss.solvers.Mixing.mean(n_contacts, sim.unit_base), 'its mean number of contacts'

    def get_edges(self, inds, n_contacts):
        """
        Efficiently find edges

        Note that each person is assigned 2N contacts (i.e. if a person has 5 contacts,
        they appear 5 times in the 'source' array and, on average, 5 times in the 'target'
        array). Therefore, the `n_contacts` argument to this function should be HALF of
        the total contacts a person is expected to have, if both the source and target
        array outputs are used (e.g. for social contacts)

        Args:
            inds (list/array): person indices
            n_contacts (list/array): the same length as `inds` with the number of bidirectional contacts to assign to each person

        Returns:
            Two arrays, for source and target
        """
        if self.pars.method == 'exact':
            return self.shuffle_edges(inds, n_contacts)
        nc = np.asarray(n_contacts)
        by_agent = self.pars.method == 'uniform' or nc.size <= 1 or (nc == nc.flat[0]).all() # With the same number of stubs each, the owner of a stub drawn at random is an agent drawn at random
        seed = self.dist.rng.integers(2**62) # The key of this step's uniforms, one for each edge (from a counter, so a large network can be made by several threads)
        source, target = ss.perflib._passes.random_edges(inds, nc, seed, by_agent) # A random target stub for each source stub; sampling stubs rather than agents keeps selection probability proportional to n_contacts
        source, target = source.view(ss.uids), target.view(ss.uids)
        self.dist.jump() # Reset the RNG manually, as in shuffle_edges()
        return source, target

    def shuffle_edges(self, inds, n_contacts):
        """ Find edges by shuffling the contacts, so each agent has exactly as many as given (`method='exact'`) """
        source = np.repeat(inds, n_contacts)
        target = source.copy()
        randvals = self.dist.rng.random(len(target)) # Precompute the random values for the shuffle
        fisher_yates_shuffle(target, randvals) # Shuffle the target in place (faster equivalent of rng.permutation)
        self.dist.jump() # Reset the RNG manually; does not auto-jump since using rng directly above # TODO, think if there's a better way
        return source, target

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
        stubs = np.repeat(inds, n_contacts)
        place = self.sim.groups.level_raw(mixing.space)[stubs]
        mobility = self.sim.groups.strats[mixing.space].mobility
        if mobility is not None and mobility.kind in ['commute', 'visit']:
            place = self.draw_places(mixing.time_share, place, self.dist.rng.random(len(stubs)))
            self.dist.jump()
        order = np.argsort(place, kind='stable')
        ends = np.cumsum(np.bincount(place, minlength=self.sim.groups.strats[mixing.space].n))
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
        """ With `dur=0` (the default), and stepping with the sim, every edge ends at the next step, before anything transmits over it, so the dead's edges need not be removed before then (see `ss.Route.one_step`) """
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
        v = nc.pars.v if isinstance(nc, ss.constant) and nc.unit is None else None
        if not ss.options.crn or not sc.isnumber(v) or not float(v).is_integer() or not n or not v:
            return None
        orig = v*n
        needed = orig/2 - current
        scaled = v*needed/orig # The same arithmetic as add_pairs(), for one agent
        return int(scaled) if needed > 0 and float(scaled).is_integer() else None

    def add_pairs(self):
        """ Generate edges """
        p = self.pars
        people = self.sim.people
        uids = self.eligible(p.age_range).uids
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

        # Get the new edges -- the key step
        p1, p2 = self.pair(uids, n_conn)
        n = len(p1)
        beta = np.full(n, self.pars.beta, dtype=ss_float)

        if isinstance(p.dur, ss.Dist):
            dur = p.dur.rvs(p1)
        elif p.dur == 0:
            dur = np.zeros(n, dtype=ss_float) # In its dtype (half the memory of float64), as validate() would make it
        else:
            dur = np.ones(n)*(p.dur/self.t.dt)

        # Use zero-copy view for UIDs and skip concatenation when network is empty
        p1 = p1.view(ss.uids)
        p2 = p2.view(ss.uids)
        if current == 0:
            self.edges.p1 = p1
            self.edges.p2 = p2
            self.edges.beta = beta
            self.edges.dur = dur
        else:
            self.append(p1=p1, p2=p2, beta=beta, dur=dur)
        return


class DegreeTargets:
    """
    Targets for the number of partners people have at a time, for `ss.SexualNet` and `ssl.DegreeNet`

    The target is a distribution of the number of partners (`degree_dist()`). Partnerships
    end at random, and people look for new ones with a probability that depends on how many
    they already have (`p_seek`), chosen so that the target is the stable state.
    """
    @staticmethod
    def degree_dist(n_contacts, concurrency=None, max_contacts=None):
        """ The target share of people with 0, 1, 2, ... partners at a time """
        if isinstance(n_contacts, ss.by):
            errormsg = f'n_contacts={n_contacts} gives each level its own number of partners, which ssl.DegreeNet takes; here give one number or distribution (on ss.SexualNet, or one for each group of between=, as a dict)'
            raise ValueError(errormsg)
        if sc.isnumber(n_contacts):
            n_contacts = ss.poisson(lam=n_contacts)
        pars = {k:v for k,v in n_contacts.pars.items()}
        top = max_contacts if max_contacts is not None else None
        if isinstance(n_contacts, ss.poisson):
            mean = pars['lam']
            k = np.arange(sc.ifelse(top, int(mean + 10*np.sqrt(mean) + 10)) + 1)
            base = sps.poisson.pmf(k, mean)
        elif isinstance(n_contacts, ss.constant):
            mean = int(pars['v'])
            k = np.arange(sc.ifelse(top, mean) + 1)
            base = (k == mean).astype(float)
        else: # Any other distribution, by sampling a copy of it
            draws = np.round(sc.dcp(n_contacts).init(force=True).rvs(100_000)).astype(int)
            mean = draws.mean()
            k = np.arange(sc.ifelse(top, draws.max()) + 1)
            base = np.bincount(np.clip(draws, 0, k[-1]), minlength=len(k)).astype(float)
        if concurrency is None and max_contacts is None:
            return base/base.sum()

        def tilt(x): # Reweight the distribution exponentially in the number of partners and in having two or more
            w = base*np.exp(x[0]*k + (x[1] if len(x) > 1 else 0)*(k >= 2))
            return w/w.sum()
        if concurrency is None:
            x = [spo.brentq(lambda a: tilt([a]) @ k - mean, -50, 50)]
        else:
            x = spo.fsolve(lambda x: [tilt(x) @ k - mean, tilt(x)[2:].sum() - concurrency], [0, 0])
        out = tilt(x)
        if abs(out @ k - mean) > 1e-6 or (concurrency is not None and abs(out[2:].sum() - concurrency) > 1e-6):
            errormsg = f'No distribution of the number of partners has a mean of {mean:n}{f", {concurrency:n} with two or more," if concurrency is not None else ""} and a maximum of {max_contacts}'
            raise ValueError(errormsg)
        return out

    @staticmethod
    def seek_probs(dist, mean_dur):
        """ The probability that someone with each number of partners looks for another on a step, for a target distribution, given how many steps partnerships last on average """
        k = np.arange(len(dist))
        ratio = np.divide(dist[1:], dist[:-1], out=np.where(dist[1:] > 0, np.inf, 0.0), where=dist[:-1] > 0) # Below the least number in the target (e.g. none with ss.constant(1)), people always look
        return np.minimum(np.append((k[:-1] + 1)*(1/mean_dur)*ratio, 0), 1) # Chosen so the target is the stationary distribution: those with k partners who gain one balance those with k+1 who lose one

    def init_seek(self, mean_dur):
        """ The probability that someone with each number of partners looks for another on a step """
        self.p_seek = self.seek_probs(self.dist_degree, mean_dur)
        return

    def start_degrees(self, uids, rng):
        """ A number of partners for each person, from the target, to start with """
        cdf = np.cumsum(self.dist_degree)
        return np.minimum(np.searchsorted(cdf, rng.rvs(uids)), len(cdf) - 1)

    def seeking(self, uids, rng):
        """ The people who look for a partner on this step """
        degree = np.minimum(self.degrees(uids), len(self.p_seek) - 1)
        return uids[rng.rvs(uids) < self.p_seek[degree]]

    def degrees(self, uids):
        """ The number of partners each person has """
        size = int(uids.max()) + 1 if len(uids) else 0
        counts = np.bincount(self.edges.p1, minlength=size)[:size] + np.bincount(self.edges.p2, minlength=size)[:size]
        return counts[uids]

    def people_seeking(self, people):
        """ The people the targets are for: see the subclasses """
        return people.alive.uids

    def levels(self, name=None):
        """ The level of every person in a property of people (by default, the one match= names), as an array indexed by UID """
        sim = self.sim
        name = list(self.pars.match.keys())[0] if name is None else name
        names = [strat.name for strat in sim.groups.strats]
        if name in names: # A stratification
            return sim.groups.level_raw(names.index(name))
        return np.asarray(sim.people[name].raw)

    def define_degree_results(self):
        self.define_results(
            ss.Result('mean_degree', label='Mean number of partners', scale=False, auto_plot=False), # Per person, so not scaled by the population
            ss.Result('concurrency', label='Share with two or more partners', scale=False, auto_plot=False),
        )
        return

    def update_degree_results(self, uids):
        degree = self.degrees(uids)
        self.results.mean_degree[self.ti] = degree.mean() if len(degree) else 0
        self.results.concurrency[self.ti] = (degree >= 2).mean() if len(degree) else 0
        return

    def check(self, n_runs=3, output=False):
        """
        Run the sim again, and compare the network it makes with the network's targets

        Reports, for each target (the mean number of partners, the share with two or
        more, the maximum, and with `match=`, the share of partnerships at the same
        level), the target, the mean over the second half of each run, and the spread
        across runs; a target that is missed by more than the spread is an `ss.errors.TargetStatsNotRecovered` warning.
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
        if not hasattr(self.sim, 'inputs'):
            errormsg = f'net.check() reruns the sim, which can only be rebuilt from the inputs it was made with; the network\'s results (mean_degree, concurrency) show how close it is to the targets'
            raise ValueError(errormsg)
        rows = []
        mixes = [] # With a number of partners for each level, the share of people at each level at the end of each run
        for i in range(n_runs):
            run = self.sim.new(rand_seed=i + 1, verbose=0).run()
            net = run.networks[self.name]
            res = net.results
            half = len(res.mean_degree)//2
            row = dict(mean_degree=res.mean_degree[half:].mean(), concurrency=res.concurrency[half:].mean(), max_contacts=net.degrees(net.people_seeking(run.people)).max())
            if 'new_partnerships' in res and getattr(net, 'mean_dur', None) is not None and res.new_partnerships[half:].mean() > 0: # Little's law: partnerships = rate formed × how long they last
                n_alive = len(run.people.alive.uids)
                row['dur'] = res.mean_degree[half:].mean()*n_alive/2/res.new_partnerships[half:].mean()
            if getattr(net, 'by_level', None) is not None:
                uids = net.people_seeking(run.people)
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
                ss.warn(f'the network "{self.name}" has {name} {row.realized:.3g}, against a target of {row.target:.3g}', ss.errors.TargetStatsNotRecovered)
        if not output:
            print(table)
        return table

    def level_names(self, name):
        """ The name of each level of a property of people, for the rows of `check()`: a stratification's labels (e.g. low, high), or else the levels' numbers """
        strats = {strat.name:strat for strat in self.sim.groups.strats}
        n = len(self.dists_degree) if getattr(self, 'dists_degree', None) is not None else 0
        return [str(label) for label in strats[name].labels] if name in strats else [str(lev) for lev in range(n)]


class SexualNet(DegreeTargets, DynamicNetwork):
    """
    Sexual partnerships that form and dissolve, between two groups of people: by default, men and women

    People take part after their debut (`debut`), if they do (`participation`, drawn at
    their debut). By default, people who have no partner are paired on every step, so
    nobody has more than one at a time. With `n_contacts`, the number of
    partners people have at a time is a target instead: partnerships end when their
    duration is up, and people look for new ones at a rate that depends on how many they
    have, so the distribution of the number of partners (among people who take part, after
    their debut) stays at the target. In an open population, where partners die and
    newcomers start with none, people look more (or less) often by a factor that keeps the
    mean number of partners at the target; with partnerships much longer than people's time
    in the network, it takes a few decades to settle. The network then starts at the
    target, and its results and `net.check()` say how close it is.

    Each partnership is between one person of each of two groups (`between`): a man and a
    woman, two men, or e.g. a female sex worker and a client, and each group can have its own
    number of partners (`n_contacts` as a dict). With `match=`, a share of partnerships is
    between people at the same level of a property of people, such as a risk group.

    Each partnership has a number of acts per step (`acts`), so a disease's beta on it is
    a probability per act (see `ss.peract`), and the share of them in which a condom is used
    (`condoms`, 0 unless set, e.g. by a subclass in `add_pairs()`), which a modifier of the
    transmission on the network reads, e.g.
    `ss.modify(hiv.infection, efficacy=0.8, share=lambda net: net.edges.condoms, route='sexualnet')`.

    A subclass that overrides `step()` forms its own partnerships: the base starts none for
    it, and has none of the parameters for forming them, only `between`.

    For several types of partnership with their own rules, see STIsim; for partnerships that
    are not sexual (e.g. needle sharing), or matching by several properties, `ssl.DegreeNet`
    (`import starsim.library as ssl`).

    Note: not CRN safe when people are born or die.

    Args:
        between (str/tuple): who partnerships are between: `'mf'` (the default), a man and a woman, so its directions are `m2f` and `f2m` and a disease's beta on it is `ss.peract(m2f=..., f2m=...)`, or one value for both; `'mm'`, two men (men who have sex with men), with one beta; or a pair of groups, each the name of a boolean property of people or of a level of a stratification, e.g. `('fsw', 'client')`, whose directions are `fsw2client` and `client2fsw` (nobody can be in both)
        dur (`ss.Dist`/`ss.dur`): the duration of each partnership; can vary by age, year, and individual pair (default: lognormal, mean 15 years, standard deviation 1 year); a single duration, e.g. `ss.months(6)`, is a mean, and partnerships end at a constant rate (an exponential), as for `ssl.DegreeNet`; give a distribution for any other spread, e.g. `ss.lognorm_ex(mean=ss.months(6), std=ss.months(3))`
        debut (`ss.Dist`): the age of debut; can vary by using callable parameter values
        acts (`ss.freq`/`ss.Dist`): the number of acts per unit time in each partnership: a frequency (e.g. `ss.freqperweek(2)`; a hazard such as `ss.peryear(80)` is an error), or a distribution of them drawn once for each partnership (default: lognormal, mean 80 a year, standard deviation 30)
        participation (`ss.Dist`): whether each person takes part, drawn at their debut; can vary by individual properties (age, sex, ...) using callable parameter values
        n_contacts (float/`ss.Dist`/dict): if given, the number of partners people have at a time: a distribution over whole numbers, or a number, which is the mean of a Poisson; or one for each group of `between`, e.g. `dict(fsw=3, client=0.5)` (each partnership has one of each, so these should give the two groups about as many partners in all)
        concurrency (float): with `n_contacts`, the share of people with two or more partners at once
        max_contacts (int): with `n_contacts`, the most partners anyone has at once
        match (dict): if given, the share of partnerships between people at the same level of a property of people (a stratification, or an attribute), e.g. `dict(risk=0.8)`

    Examples:
        ```python
        net = ss.SexualNet() # Men and women, at most one partner at a time
        msm = ss.SexualNet(name='msm', between='mm', participation=0.1, dur=ss.lognorm_ex(mean=ss.years(2), std=ss.years(1)))
        net = ss.SexualNet(n_contacts=ss.poisson(0.7), dur=ss.lognorm_ex(mean=ss.years(2), std=ss.years(1)), concurrency=0.1, match=dict(risk=0.8))
        sw = ss.SexualNet(name='sw', between=('fsw', 'client'), n_contacts=dict(fsw=5, client=0.5), dur=ss.months(1)) # With boolean attributes fsw and client
        ```
    """
    def __init__(self, pars=None, between=_, dur=_, debut=_, acts=_, participation=_, n_contacts=_, concurrency=_, max_contacts=_, match=_, **kwargs):
        super().__init__()
        self.define_pars(between='mf')
        if not self.forms_own: # A subclass that forms its own partnerships uses none of these
            self.define_pars(
                dur = ss.lognorm_ex(mean=ss.years(15), std=ss.years(1)),  # Can vary by age, year, and individual pair. Set scale=exp(mu) and s=sigma where mu,sigma are of the underlying normal distribution.
                debut = ss.normal(loc=16),  # Age of debut can vary by using callable parameter values
                acts = ss.lognorm_ex(ss.freqperyear(80), ss.freqperyear(30)), # Acts per unit time, drawn once for each partnership
                participation = ss.bernoulli(p=0.9),  # Probability of participating in this network - can vary by individual properties (age, sex, ...) using callable parameter values
                n_contacts = None,
                concurrency = None,
                max_contacts = None,
                match = None,
            )
        self.update_pars(pars, **kwargs)
        given = dur if dur is not None else (pars or {}).get('dur')
        if not self.forms_own and isinstance(given, (int, float, ss.dur)): # A single duration is a mean, and partnerships end at a constant rate, as for ssl.DegreeNet (not the default's standard deviation of a year)
            self.pars.dur = ss.expon(scale=self.pars.dur.pars.mean) # The mean set by update_pars(), in the default's unit if a bare number
        between = self.pars.between
        groups = dict(mf=('male', 'female'), mm=('male', 'male')).get(between) if isinstance(between, str) else between
        if not isinstance(groups, (tuple, list)) or len(groups) != 2 or not all(isinstance(g, str) for g in groups):
            errormsg = f'between= of "{self.name}" is "mf" (men and women), "mm" (men who have sex with men), or a pair of groups of people by name, e.g. ("fsw", "client"), not {between!r}'
            raise ValueError(errormsg)
        self.groups = tuple(groups)
        self.sides = None if groups[0] == groups[1] else {dict(male='m', female='f').get(g, g):g for g in groups} # Its directions: m2f and f2m between men and women; within one group, one beta
        self.meta.acts = ss_float # Acts in each partnership in a step: its acts per unit time times the step, a real number
        self.meta.condoms = ss_float # The share of acts with a condom, e.g. for ss.modify(..., share=lambda net: net.edges.condoms, route=)
        self.edge_defaults['condoms'] = 0.0
        self.debut = ss.FloatArr('debut') # Drawn for each person on their first step in the sim, whether born, arrived, or there at the start
        self.decided = ss.BoolArr('decided') # Whether a person's participation has been drawn, which happens at their debut, so a probability by age is read at that age
        self.rng_degree = ss.random() # The number of partners each person starts with, with n_contacts
        self.rng_seek   = ss.random() # Whether each person looks for a partner on this step, with n_contacts
        self.rng_order  = ss.random() # The order in which people are paired
        if self.pars.get('match') is not None:
            self.rng_match = ss.random() # Whether each person looks for a partner at their own level
            self.match_extra = 0.0 # How many more look within their level than by the target alone: see pair()
            self.match_results = {key:'match' for key in self.pars.match} # The result of the share at the same level
        self.waiting = ss.uids() # People who looked for a partner and were left over, who look again
        return

    @property
    def forms_own(self):
        """ Whether a subclass forms its own partnerships (it overrides `step()`), so the base forms none for it """
        return type(self).step is not SexualNet.step

    def side_raw(self):
        """ The group of `between` each person is in (-1 for neither), indexed by UID: by sex, men 0 and women 1, by default """
        if self.groups == ('male', 'female'):
            return self.sim.people.female.raw.astype(np.int8)
        elif self.sides is None:
            return np.where(people_mask(self.sim, self.groups[0]), 0, -1).astype(np.int8)
        return super().side_raw()

    @property
    def targeted(self):
        """ Whether the number of partners people have at a time is a target (`n_contacts`), rather than one at most """
        return self.pars.get('n_contacts') is not None

    def per_group(self, value):
        """ A value for each group of `between`, from a dict of them, or one value for everyone """
        if not isinstance(value, dict):
            return [value]
        if self.sides is None or set(value) != set(self.groups):
            errormsg = f'n_contacts= of "{self.name}" as a dict gives a value for each group of between= ({sc.strjoin(list(dict.fromkeys(self.groups)))}), not for {sc.strjoin(list(value))}'
            raise ValueError(errormsg)
        return [value[g] for g in self.groups]

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.forms_own:
            return
        p = self.pars
        check_acts(self, p.acts)
        if p.match is not None and (not isinstance(p.match, dict) or len(p.match) != 1 or not all(sc.isnumber(v) and 0 <= v <= 1 for v in p.match.values())):
            errormsg = f'match= of "{self.name}" takes a property of people and the share of partnerships at the same level of it, e.g. dict(risk=0.8), not {p.match}; for several, see ssl.DegreeNet'
            raise ValueError(errormsg)
        if self.targeted:
            dists = [self.degree_dist(n, p.concurrency, p.max_contacts) for n in self.per_group(p.n_contacts)]
            size = max(len(d) for d in dists)
            self.dists_degree = np.array([np.pad(d, (0, size - len(d))) for d in dists]) # One row for each group, with n_contacts by group
            self.dist_degree = self.dists_degree.mean(axis=0) # For everyone, for the results and check(): the mix of the groups, from init_post()
            mean = ss.transitions.dwell_stats(p.dur, lambda v: ss.solvers.convert(v, sim.unit_base))[0]
            self.mean_steps = mean/self.t.dt.to(sim.unit_base).value
            self.boost = np.ones(len(dists)) # How much more (or less) often people look than in a closed population, adjusted as the run goes: see add_pairs()
            self.init_seek(self.mean_steps*self.boost)
        elif p.concurrency is not None or p.max_contacts is not None:
            errormsg = f'concurrency= and max_contacts= of "{self.name}" are targets for the number of partners at a time, so need n_contacts= as well'
            raise ValueError(errormsg)
        return

    def init_seek(self, mean_steps):
        """ The probability that someone in each group with each number of partners looks for another on a step, given how many steps partnerships last on average (for each group) """
        self.p_seeks = np.array([self.seek_probs(d, m) for d,m in zip(self.dists_degree, mean_steps)])
        return

    def rows(self, uids):
        """ The row of each person in the targets: their group, with n_contacts by group (else 0 for all) """
        return self.side_raw()[uids] if len(self.dists_degree) > 1 else 0

    def seek(self, uids):
        """ The probability that each person looks for a partner on this step, given how many they have """
        return self.p_seeks[self.rows(uids), np.minimum(self.degrees(uids), self.p_seeks.shape[1] - 1)]

    def seeking(self, uids, rng):
        """ The people who look for a partner on this step """
        return uids[rng.rvs(uids) < self.seek(uids)]

    def start_degrees(self, uids, rng):
        """ A number of partners for each person, from the target of their group, to start with """
        cdfs = np.cumsum(self.dists_degree, axis=1)
        return np.minimum((cdfs[self.rows(uids)] < rng.rvs(uids)[:, None]).sum(axis=1), cdfs.shape[1] - 1)

    def init_results(self):
        super().init_results()
        if self.targeted:
            self.define_degree_results()
        for name in getattr(self, 'match_results', {}).values():
            self.define_results(ss.Result(name, label='Share of partnerships at the same level', scale=False, auto_plot=False))
        return

    def active(self, people):
        """ Return boolean array of agents who are alive, participating, past sexual debut, and in a group of `between` """
        active = self.participant & (people.age > self.debut) & people.alive
        if self.groups == ('male', 'male'):
            return active & people.male
        elif self.groups != ('male', 'female'): # Otherwise everyone is in one
            return active & (self.side_raw() >= 0)
        return active

    def available(self, people, sex):
        """ Return UIDs of active agents of the given sex who are not currently partnered. """
        available = people[sex] & self.active(people)
        available[self.edges.p1] = False
        available[self.edges.p2] = False
        return available.uids

    def people_seeking(self, people):
        """ The people who take part, after their debut """
        return self.active(people).uids

    def init_post(self, add_pairs=True):
        if self.forms_own: # Start no partnerships for a subclass that forms its own, unless with its own add_pairs()
            super().init_post(add_pairs=add_pairs and type(self).add_pairs is not SexualNet.add_pairs)
            return
        self.set_network_states()
        if not self.targeted or not add_pairs:
            super().init_post(add_pairs=add_pairs) # Pair everyone who takes part, as on each step (unless a subclass pairs its own)
            return
        super().init_post(add_pairs=False) # Start at the target: give each person who takes part a number of partners, and pair them at random
        uids = self.people_seeking(self.sim.people)
        if len(self.dists_degree) > 1: # The target for everyone, from the mix of groups
            self.dist_degree = np.bincount(self.rows(uids), minlength=len(self.dists_degree))/max(len(uids), 1) @ self.dists_degree
        else:
            self.dist_degree = self.dists_degree[0]
        stubs = np.repeat(uids, self.start_degrees(uids, self.rng_degree))
        rng = self.rng_order.rng # By position, since each person appears once for each partner
        order = rng.random(len(stubs))
        p1, p2 = self.pair(stubs, order, rng.random(len(stubs)) if self.pars.match is not None else None)
        n = len(p1)
        dur = self.pars.dur.rvs(3*n) # What is left of partnerships under way: one picked in proportion to how long it lasts, and a uniform share of it
        dur = dur[rng.choice(len(dur), size=n, p=dur/dur.sum())]*rng.random(n) if n else dur[:0]
        self.rng_order.jump()
        self.add(p1, p2, dur)
        return

    def pair(self, people, order, u=None):
        """
        Pair people in the order of a random number for each, one of each group (or two of one, as with between='mm'),
        leaving out pairs who are already partners; anyone left over waits

        With `match=`, some look for a partner at their own level (those whose `u` is under a share chosen so the
        share of partnerships at the same level is the target), and are paired within it first.
        """
        o = np.argsort(order, kind='stable')
        people = people[o]
        side = self.side_raw()[people] if self.sides is not None else np.zeros(len(people), dtype=np.int8)
        match = self.pars.match
        if match is not None and len(people):
            key, share = list(match.items())[0]
            level = np.unique(self.levels(key)[people], return_inverse=True)[1]
            frac = [np.bincount(level[side == s], minlength=level.max() + 1)/max(np.sum(side == s), 1) for s in [0, 1 if self.sides else 0]]
            by_chance = frac[0] @ frac[1] # The share at the same level if they were paired at random
            within = np.clip((share - by_chance)/(1 - by_chance) + self.match_extra, 0, 1) if by_chance < 1 else 1.0
            cell = np.where(u[o] < within, level + 1, 0) # The level each looks within, plus 1 (0 for none)
        elif match is not None:
            cell = np.zeros(0, dtype=np.int64)

        def split(idx): # Partners in turn, by position, and anyone left over
            if self.sides is None:
                n = len(idx)//2
                return idx[:n], idx[n:2*n], idx[2*n:]
            a, b = idx[side[idx] == 0], idx[side[idx] == 1]
            n = min(len(a), len(b))
            return a[:n], b[:n], np.concatenate([a[n:], b[n:]])

        if match is None:
            i1, i2, left = split(np.arange(len(people)))
        else:
            parts = [split(np.flatnonzero(cell == c)) for c in np.unique(cell[cell > 0])] # Within a level, for those who look within it
            rest = np.sort(np.concatenate([np.flatnonzero(cell == 0)] + [part[2] for part in parts])) # Then everyone else, in their order
            i1, i2, left = split(rest)
            i1, i2 = np.concatenate([part[0] for part in parts] + [i1]), np.concatenate([part[1] for part in parts] + [i2])
            if len(i1): # People left over without a partner at their level make the share at the same level fall short, so more look within it, smoothed over about 20 steps
                self.match_extra = np.clip(self.match_extra + 0.05*(share - np.mean(level[i1] == level[i2])), -1, 1)
        p1, p2 = people[i1], people[i2]
        self.waiting = people[left].unique() # Otherwise the surplus of one group would be lost on every step
        keep = p1 != p2
        if len(self) and len(p1) and self.targeted: # Without targets, only people with no partner look
            existing = set(zip(np.minimum(self.edges.p1, self.edges.p2).tolist(), np.maximum(self.edges.p1, self.edges.p2).tolist()))
            keep &= np.array([pair not in existing for pair in zip(np.minimum(p1, p2).tolist(), np.maximum(p1, p2).tolist())], dtype=bool)
        return p1[keep], p2[keep]

    def add(self, p1, p2, dur=None):
        """ Add partnerships, with their durations (drawn if not given) and acts """
        if len(p1):
            by = p1 if len(np.unique(p1)) == len(p1) else len(p1) # Drawn for each person, unless someone forms two at once (they would get the same)
            dur = self.pars.dur.rvs(by) if dur is None else dur
            self.append(p1=p1, p2=p2, beta=np.ones(len(p1)), dur=dur, acts=self.draw_acts(by))
        return len(p1)

    def draw_acts(self, n):
        """ The acts in a step of each new partnership (for some people, or a number of them): from its acts per unit time, the same for all if a rate, or drawn once if a distribution """
        acts = self.pars.acts
        if isinstance(acts, ss.Dist):
            return acts.rvs(n).astype(ss_float)
        size = n if sc.isnumber(n) else len(n)
        return np.full(size, ss.solvers.convert(acts, self.sim.unit_base)*self.t.dt.to(self.sim.unit_base).value, dtype=ss_float)

    def set_network_states(self):
        """ Draw the age of debut of anyone who has none (newborns, and people who arrive at any age), and whether each person takes part, when they reach their debut """
        people = self.sim.people
        debuts = lambda reached: ss.perflib._passes.debuts(people.auids, people.alive.raw, self.decided.raw, people.age.raw, self.debut.raw, reached).view(ss.uids)
        uids = debuts(False) # Alive, with no debut age
        self.debut[uids] = self.pars.debut.rvs(uids)
        uids = debuts(True) # Alive, at least their debut age, and not decided
        self.participant[uids] = self.pars.participation.rvs(uids)
        self.decided[uids] = True
        return

    def add_pairs(self):
        """ Pair the people looking for a partner: everyone who takes part and has none, or with n_contacts, those who look on this step, given how many they have """
        active = self.people_seeking(self.sim.people)
        if not self.targeted:
            partnered = np.zeros(len(self.sim.people.alive.raw), dtype=bool)
            partnered[self.edges.p1] = True
            partnered[self.edges.p2] = True
            seekers = active[~partnered[active]]
        else:
            seekers = self.seeking(active, self.rng_seek)
            waiting = self.waiting[np.isin(self.waiting, active)]
            waiting = waiting[self.seek(waiting) > 0] # Unless they have as many partners as anyone can now
            seekers = ss.uids.concatenate([seekers, waiting]).unique()
        p1, p2 = self.pair(seekers, self.rng_order.rvs(seekers), self.rng_match.rvs(seekers) if self.pars.match is not None else None)
        n = self.add(p1, p2)
        if self.targeted and len(active): # In an open population, partners leave and newcomers start with none, so people look more (or less) often, by as much as keeps the mean number of partners at the target (a factor adjusted by a fifth of the gap each step, on a log scale), in each group
            target = self.dists_degree @ np.arange(self.dists_degree.shape[1])
            degree = self.degrees(active)
            rows = self.rows(active) if len(target) > 1 else None
            mean = np.array([degree.mean()]) if len(target) == 1 else np.array([degree[rows == r].mean() if np.any(rows == r) else t for r,t in enumerate(target)])
            gap = np.divide(target, np.maximum(mean, 1e-3*target), out=np.ones(len(target)), where=target > 0)
            self.boost = np.clip(self.boost*gap**0.5, 0.01, 1000.0)
            self.init_seek(self.mean_steps/(self.boost*gap)) # The gap itself too, which damps the swings of the slow adjustment
        return n

    def step(self):
        self.end_pairs()
        self.set_network_states() # Debut for anyone new, and participation for anyone who has reached their debut
        self.add_pairs()
        return

    def update_results(self):
        super().update_results()
        if self.targeted:
            self.update_degree_results(self.people_seeking(self.sim.people))
        if len(self):
            for key,name in getattr(self, 'match_results', {}).items():
                level = self.levels(key)
                self.results[name][self.ti] = np.mean(level[self.edges.p1] == level[self.edges.p2])
        return


class PrenatalNet(Network):
    """
    Prenatal transmission network

    Although edges change over time, the edges are managed by ss.Pregnancy as pregnancies start and finish (with fetuses as agents).
    The prenatal network therefore is not derived from DynamicNetwork as it does not manage its own updates.
    It transmits one way only, from the mother (`p1`) to the child (`p2`), so its one direction is `mother2child`.

    Note: CRN safe.
    """
    sides = dict(mother='p1', child='p2') # Where they are in an edge
    one_way = True

    def step(self):
        # Adding/removing edges is managed by ss.Pregnancy
        pass

    def add_pairs(self, mother_uids=None, unborn_uids=None):
        n = len(mother_uids) if mother_uids is not None else 0
        if n:
            self.append(p1=mother_uids, p2=unborn_uids, beta=np.ones(n))
        return n


class MaternalNet(PrenatalNet):
    """
    Mother-to-child network, for when prenatal and postnatal transmission need not be told apart

    For now the same as `ss.PrenatalNet`, but named `maternalnet`.

    Note: CRN safe.
    """
    pass


class PostnatalNet(DynamicNetwork):
    """
    Network to track postnatal processes

    This network automatically has edges added between mothers and infants upon birth if
    a `Pregnancy` module is present in the simulation. The edges persist until the specified
    duration has elapsed, at which point they are automatically removed. This allows
    different `PostnatalNet` instances have different postnatal periods specific to
    what is being modelled. Transmission along this network is possible if a beta
    value >0 is specified. Otherwise, transmission will be skipped.

    By default, the postnatal duration is based on the maternal UID and would be the same
    for all infants of that mother, although derived classes could modify this behavior.
    It transmits one way only, from the mother (`p1`) to the child (`p2`), so its one direction is `mother2child`.

    Note: CRN safe.

    Args:
        pars (dict): passed to `ss.DynamicNetwork`
        dur (ss.dur/ss.Dist): edge duration (postpartum period); if None, edges persist until agents die
        **kwargs (dict): merged with `pars`
    """

    sides = dict(mother='p1', child='p2') # Where they are in an edge
    one_way = True

    def __init__(self, pars=None, dur=None, **kwargs):
        super().__init__()
        self.define_pars(dur=dur)
        self.update_pars(pars, **kwargs)
        return

    def add_pairs(self, mother_uids=None, infant_uids=None):
        # Note that the mother_uids should contain repeat entries if a mother has multiple infants
        n = len(mother_uids) if mother_uids is not None else 0
        p = self.pars

        if n:
            if isinstance(p.dur, ss.Dist):
                dur = p.dur.rvs(mother_uids) + 1 # +1 since edges are added by Pregnancy before end_pairs() decrements them in the same step
            elif p.dur is None:
                dur = np.full(n,fill_value=np.inf)
            else:
                dur = np.full(n, p.dur/self.t.dt + 1)
            self.append(p1=mother_uids, p2=infant_uids, beta=np.ones(n), dur=dur)
        return n


class BreastfeedingNet(PostnatalNet):
    """
    Network for breastfeeding transmission

    To model breastfeeding transmission, include this network in a `Sim` that contains a `Pregnancy`
    module. Edges will be automatically added and removed from this network tracking the breastfeeding
    status from the `Pregnancy` module (`preg.lactating`).

    Note: CRN safe.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(pars=pars, **kwargs)
        assert self.pars.dur is None, 'BreastfeedingNet does not accept a `dur` parameter as breastfeeding duration is determined by the Pregnancy module'
        return

    def init_pre(self, sim):
        """ Find the sim's ss.Pregnancy, which pairs each newborn with a mother who breastfeeds (see `ss.Pregnancy.lactate()`) """
        super().init_pre(sim)
        pregs = [mod for mod in self.sim.modules if isinstance(mod, ss.Pregnancy)]
        if len(pregs) != 1:
            raise RuntimeError(f'BreastfeedingNet requires one ss.Pregnancy, which tracks breastfeeding; the sim has {len(pregs)}')
        self.pregnancy = pregs[0]
        return

    def end_pairs(self):
        """ Remove edges for mothers who have stopped breastfeeding (`preg.lactating`) """
        active = self.pregnancy.lactating[self.p1]
        for k in self.meta_keys():
            self.edges[k] = self.edges[k][active]
        return np.count_nonzero(active)


#%% Mixing pools



class AgeGroup(sc.prettyobj):
    """ A simple age-based filter that returns uids of agents that match the criteria """
    def __init__(self, low, high, do_cache=True):
        self.low = low
        self.high = high

        self.do_cache = do_cache
        self.uids = None # Cached
        self.ti_cache = -1

        self.name = repr(self)
        return

    def __call__(self, sim):
        if (not self.do_cache) or (self.ti_cache != sim.ti):
            in_group = sim.people.age >= self.low
            if self.high is not None:
                in_group = in_group & (sim.people.age < self.high)
            self.uids = ss.uids(in_group)
            self.ti_cache = sim.ti
        return self.uids

    def __repr__(self):
        return f'age({self.low}-{self.high})'


class LevelGroup(sc.prettyobj):
    """ A filter that returns the uids of the agents at given levels of the sim's stratifications, e.g. `LevelGroup(age=0, location=1)` """
    def __init__(self, **levels):
        self.levels = levels
        self.name = repr(self)
        return

    def __call__(self, sim):
        in_group = sim.people.alive
        for dim,level in self.levels.items():
            strat = sim.groups.strats[sim.groups.dim_index(dim)]
            raw = sim.people[strat.attr]
            in_group = in_group & ((raw >= strat.edges[level]) & (raw < np.append(strat.edges, np.inf)[level+1]) if strat.edges is not None else (raw == level))
        return ss.uids(in_group)

    def __repr__(self):
        return ','.join(f'{dim}={level}' for dim,level in self.levels.items())


class MixingPools(Route):
    """
    A container for creating a rectangular array of MixingPool instances

    By default, separates the population into <15 and >15 age groups.

    Note: CRN safe as agents.

    Args:
        diseases (str): the diseases that transmit via these mixing pools
        src (inds): source agents; can be AgeGroup(), ss.uids(), or lambda(sim); None indicates all alive agents
        dst (inds): destination agents; as above
        beta (float): overall transmission via these mixing pools
        n_contacts (array): the relative connectivity between different mixing pools (can be float or Dist)

    Examples:
        ```python
        import starsim as ss
        mps = ss.MixingPools(
            diseases = 'sis',
            beta = 0.1,
            src = {'0-15': ss.AgeGroup(0, 15), '15+': ss.AgeGroup(15, None)},
            dst = {'0-15': ss.AgeGroup(0, 15), '15+': ss.AgeGroup(15, None)},
            n_contacts = [[2.4, 0.49], [0.91, 0.16]],  # Illustrative contact matrix
        )
        sim = ss.Sim(diseases='sis', networks=mps).run()
        sim.plot()
        ```
    """
    def __init__(self, pars=None, diseases=_, src=_, dst=_, beta=_, n_contacts=_, **kwargs):
        super().__init__()
        self.define_pars(
            diseases = None,
            src = None,
            dst = None,
            beta = 1.0,
            n_contacts = None,
        )
        self.update_pars(pars, **kwargs)
        self.validate_pars()
        self.pools = []
        self.constant = False # Whether a number of contacts is the same for everyone in a group (as from ss.contacts()), rather than Poisson
        self.edges = None # The lower edges of the matrix's own age bands (from ss.contacts()), whose exposure agents work out by band: see band_exposure()
        self.unit = None # The unit of time the contacts are counted over, if not the sim's (e.g. 'day' for ss.contacts('Kenya')): see to_unit()
        return

    def to_unit(self, sim):
        """ Put the contacts in the sim's unit, once: a matrix counted over a unit of its own (`ss.contacts('Kenya')`, contacts a day) is scaled to contacts per unit of the sim's time """
        if self.unit is not None:
            factor = ss.solvers.convert(getattr(ss, f'per{self.unit.removesuffix("s")}')(1.0), sim.unit_base)
            self.pars.n_contacts = self.pars.n_contacts*factor
            if getattr(self, 'dims', None) is not None:
                self.dims = {dim:mat*factor for dim,mat in self.dims.items()}
            if getattr(self, 'full', None) is not None:
                self.full.matrix = self.full.matrix*factor
            self.unit = None
        return

    def __len__(self):
        try:    return len(self.pools)
        except: return 0

    def to_spec(self):
        """ The mixing pools' part of the model spec (see `ss.Module.to_spec()`): a contact matrix from `ss.contacts()` as its numbers, or a named one as where it came from """
        out = super().to_spec()
        ref = getattr(self, 'ref', None)
        if ref is not None: # Named data, e.g. ss.contacts('Kenya'): the parameters that are single values (the rest come from the data)
            out.pars = {k:v for k,v in out.pars.items() if self.pars[k] is None or isinstance(self.pars[k], (str, bool, int, float, ss.dur, ss.Rate, ss.Dist))}
            out.data = sc.jsonify(ref)
        if getattr(self, 'dims', None) is not None: # A matrix over each of the model's stratifications
            out.contacts = {dim:mat.tolist() for dim,mat in self.dims.items()}
            if getattr(self, 'age_edges', None) is not None: # The one by age with its own age groups
                out.contacts = dict(ages=self.age_edges.tolist(), matrix=out.contacts.pop('age')) | out.contacts
        elif getattr(self, 'full', None) is not None: # One matrix over several stratifications together
            out.contacts = dict(matrix=self.full.matrix.tolist(), dims=self.full.dims)
        elif getattr(self, 'ages', None) is not None: # A matrix with its own age groups
            out.contacts = dict(matrix=self.pars.n_contacts.T.tolist(), ages=self.ages)
        if self.unit is not None and ref is None: # Not yet put in the sim's unit (a named one says its own)
            out.contacts_unit = self.unit
        return out

    def validate_pars(self):
        """ Check that src and dst have correct types, and contacts is the correct shape """
        p = self.pars

        # Validate src and dst
        if not isinstance(p.src, dict):
            errormsg = f'src must be a provided as a dictionary, not {type(self.pars.src)}'
            raise TypeError(errormsg)
        if not isinstance(p.dst, dict):
            errormsg = f'dst must be a provided as a dictionary, not {type(p.dst)}'
            raise TypeError(errormsg)
        p.src = sc.objdict(p.src)
        p.dst = sc.objdict(p.dst)

        # Validate the n_contacts
        if p.n_contacts is None:
            p.n_contacts = np.ones((len(p.src), len(p.dst)))
        p.n_contacts = np.array(p.n_contacts)
        actual = p.n_contacts.shape
        expected = (len(p.src), len(p.dst))
        if actual != expected:
            errormsg = f'The number of source and destination groups must match the number of rows and columns in the mixing matrix, but {actual} != {expected}.'
            raise ValueError(errormsg)

        return

    def init_full(self, sim):
        """ For one matrix over several stratifications together (`ss.contacts(matrix, dims=[...])`), a group for each combination of their levels, the first slowest; checks the shape against the levels """
        full = self.full
        names = [strat.name for strat in sim.groups.strats]
        for dim in full.dims:
            if dim not in names:
                raise ss.errors.MissingDimension(f'the contacts "{self.name}" are given by "{dim}", but the model is not stratified by it. Add e.g. sim.stratify({dim}=[...]).')
        strats = [sim.groups.strats[names.index(dim)] for dim in full.dims]
        counts = tuple(strat.n for strat in strats)
        if (full.levels is not None and tuple(full.levels) != counts) or len(full.matrix) != np.prod(counts):
            levels = '; '.join(f'"{strat.name}" has {strat.n} ({sc.strjoin(strat.labels)})' for strat in strats)
            shape = full.matrix.shape if full.levels is None else tuple(full.levels)*2
            raise ss.errors.ShapeError(f'the contact matrix of "{self.name}" has shape {shape}, but {levels}, so it needs shape {counts*2} (or {(int(np.prod(counts)),)*2}, the first stratification slowest)')
        groups = {}
        for combo in itertools.product(*[range(n) for n in counts]):
            group = LevelGroup(**dict(zip(full.dims, combo)))
            groups[group.name] = group
        p = self.pars
        p.src, p.dst = sc.objdict(groups), sc.objdict(sc.dcp(groups))
        p.n_contacts = full.matrix.T # Indexed [source, destination]
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.to_unit(sim)
        p = self.pars
        time_args = {k:p.get(k) for k in ss.Timeline.time_args} # get() allows None
        if getattr(self, 'full', None) is not None:
            self.init_full(sim)

        self.pools = []
        coded = any(isinstance(dis, ss.Infection) for dis in sim.pars.diseases.values()) # A disease written as code transmits through the pools themselves
        if (self.name in sim.mixing.pool_terms or self.edges is not None) and not coded: # Declared diseases use the matrix from the counts in each group or band: see ss.solvers.Mixing.init_pools() and band_exposure()
            return
        if getattr(self, 'age_edges', None) is not None: # The pools of a disease written as code are by the levels of each dimension, so they need the age matrix over the model's age groups
            ages = next((strat.edges for strat in sim.groups.strats if strat.kind == 'age'), None)
            if ages is None or len(ages) != len(self.age_edges) or not np.allclose(ages, self.age_edges):
                raise ss.CapabilityError(f'the contacts "{self.name}" have age groups of their own ({sc.strjoin(self.age_edges)}) beside other dimensions, which a disease written as code cannot use yet; stratify the sim by the same age groups (stratify=dict(age={self.age_edges.tolist()}))')
        for i,sk,src in p.src.enumitems():
            for j,dk,dst in p.dst.enumitems():
                n_contacts = p.n_contacts[i,j]
                if sc.isnumber(n_contacts): # If it's a number, convert to a distribution
                    n_contacts = ss.constant(n_contacts) if self.constant else ss.poisson(lam=n_contacts)
                name = f'{self.name}:{sk}->{dk}' # Named after the whole, so that two of them (e.g. home and school contacts) can be in one sim
                mp = MixingPool(name=name, diseases=p.diseases, beta=p.beta, n_contacts=n_contacts, src=src, dst=dst, **time_args)
                mp.init_pre(sim) # Initialize the pool
                self.pools.append(mp)
        return

    def init_post(self):
        """ Initialize each mixing pool """
        super().init_post()
        for mp in self.pools:
            mp.init_post()
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
        band = np.searchsorted(self.edges, ppl.age.raw[auids], side='right') - 1
        ok = band >= 0
        nb = len(self.edges)
        n = np.bincount(band[ok], minlength=nb)
        infectious = np.bincount(band[ok], weights=weights[auids][ok], minlength=nb)
        prev = np.divide(infectious, n, out=np.zeros(nb), where=n > 0)
        expo = np.zeros(len(auids))
        expo[ok] = (self.pars.n_contacts.T @ prev)[band[ok]] # n_contacts is indexed [source, destination]
        return auids, self.pars.beta*expo

    def compute_transmission(self, *args, **kwargs):
        new_cases = []
        for mp in self.pools:
            new_cases.extend(mp.compute_transmission(*args, **kwargs))
        return new_cases

    def remove_uids(self, uids):
        """ Remove UIDs from each mixing pool """
        for mp in self.pools:
            mp.remove_uids(uids)
        return

    def step(self):
        for mp in self.pools:
            mp.step()
        return

    def shrink(self):
        """ Shrink the size of the mixing pools for saving to disk """
        super().shrink()
        for mp in self.pools:
            mp.shrink()
        return


def transmits(disease):
    """ Whether a disease is communicable: declared with a contact-mediated transition, or written as code with a beta """
    if disease.graph is not None:
        return len(disease.graph.transmissions) > 0
    return 'beta' in disease.pars


class MixingPool(Route):
    """
    Define a single mixing pool; can be used as a drop-in replacement for a network.

    Note: CRN safe as agents.

    Args:
        diseases (str): the diseases that transmit via this mixing pool
        src (inds): source agents; can be AgeGroup(), ss.uids(), or lambda(sim); None indicates all alive agents
        dst (inds): destination agents; as above
        beta (float): overall transmission (note: use a float, not a TimePar; the time component is usually handled by the disease beta)
        n_contacts (Dist): the number of effective contacts of the destination agents

    Examples:
        ```python
        import starsim as ss

        # Set the parameters
        mp_pars = dict(
            src = lambda sim: sim.people.male, # only males are infectious
            dst = None, # all agents are susceptible
            beta = 0.2,
            n_contacts = ss.poisson(lam=4),
        )

        # Seed 5% of the male population
        def p_init(self, sim, uids):
            return 0.05*sim.people.male

        # Create and run the sim
        sis = ss.SIS(init_prev=p_init)
        mp = ss.MixingPool(**mp_pars)
        sim = ss.Sim(diseases=sis, networks=mp)
        sim.run()
        sim.plot()
        ```
    """
    def __init__(self, pars=None, diseases=_, src=_, dst=_, beta=_, n_contacts=_, **kwargs):
        super().__init__()
        self.define_pars(
            diseases = None,
            src = None,
            dst = None, # Same as src
            beta = 1.0,
            n_contacts = ss.constant(10),
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.FloatArr('eff_contacts', default=self.pars.n_contacts, label='Effective number of contacts')
        )

        self.pars.diseases = sc.tolist(self.pars.diseases)
        self.diseases = None
        self.src_uids = None
        self.dst_uids = None
        self.p_acquire = ss.bernoulli(p=0) # Placeholder value
        return

    def shrink(self):
        """ Shrink the size of the mixing pool for saving to disk """
        super().shrink()
        shrunk = ss.utils.shrink()
        self.diseases = shrunk
        self.src_uids = shrunk
        self.dst_uids = shrunk
        self.setattribute('t', shrunk) # Its own copy of the timeline, for each of what can be many pools
        return

    def __len__(self):
        try:
            return len(self.pars.dst)
        except:
            return 0

    def init_pre(self, sim):
        """ A number of contacts given as a rate (e.g. `ss.perday(10)`) is a number per unit of the sim's time, as a bare number is """
        n = self.pars.n_contacts
        rate = n.pars.v if isinstance(n, ss.constant) else n
        if isinstance(rate, ss.Rate):
            self.pars.n_contacts = ss.constant(ss.solvers.convert(rate, sim.unit_base))
            self.eff_contacts.default = self.pars.n_contacts
        super().init_pre(sim)
        return

    def init_post(self):
        super().init_post()

        if len(self.pars.diseases) == 0:
            self.diseases = [d for d in self.sim.diseases.values() if transmits(d)] # Assume the user wants all communicable diseases
        else:
            self.diseases = []
            for d in self.pars.diseases:
                if not isinstance(d, str):
                    raise TypeError(f'Diseases can be specified as disease objects or strings, not {type(d)}')
                if d not in self.sim.diseases:
                    raise KeyError(f'Could not find disease with name {d} in the list of diseases.')

                dis = self.sim.diseases[d]
                if not transmits(dis):
                    raise TypeError(f'Cannot create a mixing pool for disease {d}. Mixing pools only work for communicable diseases.')
                self.diseases.append(dis)

            if len(self.diseases) == 0:
                raise ValueError('You must specify at least one transmissible disease to use mixing pools')
        return

    def get_uids(self, func_or_array):
        if func_or_array is None:
            return self.sim.people.auids
        elif callable(func_or_array):
            out = func_or_array(self.sim)
            return out.uids if isinstance(out, ss.BoolArr) else out # A boolean array (e.g. sim.people.female) is over the agents alive, so is converted to their UIDs
        elif isinstance(func_or_array, ss.uids):
            return func_or_array
        errormsg = f'src must be either a callable function, e.g. lambda sim: ss.uids(sim.people.age<5), or an array of uids, not {type(func_or_array)}'
        raise TypeError(errormsg)

    def remove_uids(self, uids):
        """ If UIDs are supplied explicitly, remove them if people die """
        for key in ['src', 'dst']:
            inds = self.pars[key]
            if isinstance(inds, ss.uids):
                self.pars[key] = inds.remove(uids)
        return

    def compute_transmission(self, rel_sus, rel_trans, disease_beta, disease):
        """
        Calculate transmission

        This is called from Infection.transmit() together with network transmission.

        Args:
            rel_sus (float): Relative susceptibility
            rel_trans (float): Relative infectiousness
            disease_beta (float): The beta value for the disease
        Returns:
            UIDs of agents who acquired the disease at this step
        """
        if (disease_beta == 0) or (disease not in self.diseases):
            return []

        # Determine the mixing pool beta value
        beta = self.pars.beta
        if isinstance(beta, ss.Rate):
            ss.warn('In mixing pools, beta should typically be a float')
            beta = beta.to_prob(self.t.dt)
        if sc.isnumber(beta) and beta == 0:
            return []

        if len(self.src_uids) == 0 or len(self.dst_uids) == 0:
            return []

        # Calculate transmission
        trans = np.mean(rel_trans[self.src_uids])
        acq = self.eff_contacts[self.dst_uids] * rel_sus[self.dst_uids]
        p = beta*disease_beta*trans*acq
        self.p_acquire.set(p=p)
        return self.p_acquire.filter(self.dst_uids)

    def step(self):
        """ Update source and target UIDs """
        self.src_uids = self.get_uids(self.pars.src)
        self.dst_uids = self.get_uids(self.pars.dst)
        return



def check_contacts(matrix, dim):
    """ The entries of a contact matrix are numbers of contacts, so are finite and not negative """
    bad = ~np.isfinite(matrix) | (matrix < 0)
    if bad.any():
        i, j = np.argwhere(bad)[0]
        errormsg = f'The contact matrix of ss.contacts() for "{dim}" has {bad.sum()} entries that are negative or not finite, e.g. {matrix[i, j]} in row {i}, column {j}; each is a number of contacts per unit time'
        raise ValueError(errormsg)
    return


def contacts(matrix=None, ages=None, dims=None, name=None, source=None, layer='all', unit=None):
    """
    Mixing between groups from a contact matrix; returns `ss.MixingPools`

    A matrix, and what it is over:

    - a country's, by name (`ss.contacts('Kenya')`), from `starsim_data`, by 5-year age group;
    - its own age groups, given by their lower edges (`ss.contacts(matrix, ages=[0, 15, 65])`), which need not be the model's (it is regrouped to them);
    - the model's own stratifications: its age groups (`ss.contacts(matrix)`), another one (`ss.contacts(matrix, dims='risk')`), or several together
      (`ss.contacts(matrix, dims=['sex', 'risk'])`), for mixing that is not a product, e.g. men mix with women, and by risk differently for each sex.

    Mixing that is a product over stratifications takes a matrix for each, by name (`ss.contacts(dict(age=..., area=...))`): the contacts between
    two groups are the product of the entries for each. The matrix by age can have its own age groups here too (`ss.contacts(dict(age=..., area=...), ages=[0, 15, 65])`).

    Args:
        matrix (array/str/dict): `matrix[i,j]`, the number of contacts per unit time that one person in group `i` has with people in group `j`; or a country, by name or code ('Kenya', 'KE'), or a US state ('United States:Massachusetts'); or a matrix for each of the model's stratifications, by name
        ages (list/array): the lower edge of each age group of the matrix (or of its matrix by age), e.g. `[0, 15, 65]` for 0–14, 15–64, and 65+
        dims (str/list): the model's stratifications the matrix is over (default: age); for several, `matrix[i, k, j, l]` is the number of contacts per unit time
            that one person at levels `(i, k)` has with people at levels `(j, l)`, or the same as a square matrix over the combinations of levels, the first stratification slowest
        name (str): the name of the mixing (default: 'contacts', or the layer, e.g. 'home')
        source (str): for a country, the study: 'prem2021' (the default; bundled, 177 countries), 'prem2017' (152), or 'mistry2021' (36, by single year of age); for a US state, 'mistry2021_us' (the default; by single year of age); all but the first are downloaded on first use
        layer (str/list): for a country, the setting: 'all' (the default), 'home', 'school', 'work', or 'community'; or a list of them, for a list of layers, each named by its setting
        unit (str): the unit of time the contacts are counted over, e.g. 'day' for contacts a day; by default 'day' for a country's matrix (as the studies give them), and otherwise the sim's unit, as for any bare number

    Examples:
        ```python
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ 0.05*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01)
        mixing = ss.contacts([[10, 3], [3, 5]], ages=[0, 18])
        sim = ss.Sim(sir, networks=mixing, unit='day', stop=100).run()

        sir.stratify(risk=['low', 'high'])
        mixing = ss.contacts([[2, 1], [1, 8]], dims='risk')

        home, school = ss.contacts('Kenya', layer=['home', 'school']) # Contacts a day, by 5-year age group
        sim = ss.Sim(sir, networks=[home, school], stratify=dict(age=[0, 5, 18, 65]), unit='day', stop=100)

        # A product: by age, then 90% of contacts within one's own area
        mixing = ss.contacts(dict(age=[[10, 3], [3, 5]], area=[[0.9, 0.1], [0.1, 0.9]]), ages=[0, 18])

        # Men mix with women, and the high-risk with each other more among men than among women: a matrix over sex and risk together
        matrix = np.zeros((2, 2, 2, 2)) # [sex, risk, sex, risk], with sex ['female', 'male'] and risk ['low', 'high']
        matrix[0, :, 1, :] = [[0.8, 0.2], [0.5, 1.5]] # Women's contacts with men
        matrix[1, :, 0, :] = [[0.8, 0.5], [0.2, 2.0]] # Men's contacts with women
        mixing = ss.contacts(matrix, dims=['sex', 'risk'])
        ```
    """
    if isinstance(layer, (list, tuple)): # Several settings, e.g. ss.contacts('Kenya', layer=['home', 'school', 'work', 'community'])
        return [contacts(matrix, ages=ages, dims=dims, source=source, layer=lay, unit=unit) for lay in layer]
    name = name or (layer if layer != 'all' else 'contacts')
    if dims is not None and (isinstance(matrix, (str, dict)) or ages is not None):
        errormsg = f'ss.contacts(matrix, dims={dims!r}) is a matrix over the model\'s own stratifications, so it takes no ages= (a matrix with its own age groups is ss.contacts(matrix, ages=[...])), and a matrix, not a name or a dict'
        raise ValueError(errormsg)
    if dims is not None and len(sc.tolist(dims)) > 1: # One matrix over several of the model's stratifications together
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
        out = MixingPools(src={}, dst={}, n_contacts=np.zeros((0, 0)), name=name) # Its pools are made when the sim is built, once the levels are known (see MixingPools.init_full())
        out.constant = True
        out.full = sc.objdict(dims=over, matrix=matrix, levels=levels) # So a compartmental run can use the matrix directly
        out.unit = unit
        return out
    if dims is not None: # A matrix over one of the model's stratifications
        matrix = {sc.tolist(dims)[0]:matrix}
    elif matrix is not None and not isinstance(matrix, (str, dict)) and ages is None: # Over the model's own age groups
        matrix = dict(age=matrix)
    age_edges = None
    if isinstance(matrix, dict) and ages is not None: # A matrix by age with its own age groups, with a matrix for each of other dimensions: regrouped to the model's age groups when the sim is built
        if 'age' not in matrix:
            errormsg = f'ss.contacts(dict(...), ages=...) gives the age groups of the matrix by age, but there is none, only {sc.strjoin(matrix.keys())}; e.g. ss.contacts(dict(age=[[...]], {list(matrix)[0]}=[[...]]), ages=[0, 15, 65])'
            raise ValueError(errormsg)
        if len(matrix) == 1: # Only by age: a matrix with its own age groups
            matrix = matrix['age']
        else:
            mat = np.array(matrix['age'], dtype=float)
            if len(ages) != len(mat) or mat.ndim != 2 or mat.shape[0] != mat.shape[1] or np.any(np.diff(np.array(ages, dtype=float)) <= 0):
                errormsg = f'ss.contacts() needs a square matrix and the lower edge of each of its age groups, increasing, e.g. ss.contacts(matrix, ages=[0, 15, 65]); got a matrix of shape {mat.shape} and ages={list(ages)}'
                raise ValueError(errormsg)
            age_edges = np.asarray(ages) # As given (e.g. whole numbers), for the model spec
            ages = None
    if isinstance(matrix, dict): # A matrix for each of the model's own dimensions
        mats = {dim:np.array(mat, dtype=float) for dim,mat in matrix.items()}
        for dim,mat in mats.items():
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                errormsg = f'The contact matrix of ss.contacts() for "{dim}" needs to be square, with a row and a column for each level of "{dim}"; got shape {mat.shape}'
                raise ValueError(errormsg)
            check_contacts(mat, dim)
        levels = list(itertools.product(*[range(len(mat)) for mat in mats.values()]))
        groups = {}
        for combo in levels:
            group = LevelGroup(**dict(zip(mats.keys(), combo)))
            groups[group.name] = group
        product = functools.reduce(np.kron, mats.values(), np.ones((1, 1))) # The product of the entries for each dimension, for each pair of combinations of levels (in the order of `levels`)
        out = MixingPools(src=groups, dst=sc.dcp(groups), n_contacts=product.T, name=name) # ss.MixingPools is indexed [source, destination]; kept as numbers, since a matrix with many levels would be a million objects to make and copy
        out.constant = True # Everyone in a group has the same number of contacts
        out.dims = mats # So a compartmental run can use the matrices directly
        out.age_edges = age_edges # The lower edges of the age matrix's own groups, if given with ages= (see ss.solvers.Mixing.dim_matrices())
        out.unit = unit
        return out
    ref = None
    if isinstance(matrix, str): # A country's matrix, from starsim_data
        matrix, ages, ref = ss.data.contacts(matrix, source=source, layer=layer)
    elif matrix is None:
        errormsg = 'ss.contacts() needs a matrix: a country (ss.contacts("Kenya")), a matrix with its own age groups (ss.contacts(matrix, ages=[...])), or one over the model\'s stratifications (ss.contacts(matrix), ss.contacts(matrix, dims=\'risk\'))'
        raise ValueError(errormsg)
    matrix = np.array(matrix, dtype=float)
    if ages is None or len(ages) != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        errormsg = f'ss.contacts() needs a square matrix and the lower edge of each of its age groups, e.g. ss.contacts(matrix, ages=[0, 15, 65]); got a matrix of shape {matrix.shape} and ages={ages}'
        raise ValueError(errormsg)
    check_contacts(matrix, 'age')
    if np.any(np.diff(np.array(ages, dtype=float)) <= 0):
        errormsg = f'The age groups of ss.contacts() are given by their lower edges, which need to increase, e.g. ages=[0, 15, 65]; got ages={list(ages)}'
        raise ValueError(errormsg)
    highs = list(ages[1:]) + [None]
    groups = {(f'{lo}-{hi}' if hi is not None else f'{lo}+'): AgeGroup(lo, hi) for lo,hi in zip(ages, highs)}
    out = MixingPools(src=groups, dst=sc.dcp(groups), n_contacts=matrix.T, name=name) # ss.MixingPools is indexed [source, destination]
    out.constant = True # Everyone in a group has the same number of contacts
    out.ref = ref # Where a named matrix came from, for the model spec
    out.ages = None if ref is not None else np.asarray(ages).tolist() # A matrix given with its own age groups, for the model spec (as given, since they name the groups)
    out.edges = np.asarray(ages, dtype=float)
    out.unit = unit or ('day' if ref is not None else None) # S7.8: a country's matrix is contacts a day, whatever the sim's unit
    return out
