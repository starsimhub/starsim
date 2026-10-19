"""
Networks that connect people within a population
"""
import inspect
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



# %% General network classes

class Route(ss.Module):
    """
    A transmission route -- e.g., a network, mixing pool, environmental transmission, etc.
    """
    def compute_transmission(self, rel_sus, rel_trans, disease_beta, disease=None):
        """ Compute transmission probabilities along this route (must be overridden by subclasses). """
        errormsg = 'compute_transmission() must be defined by Route subclasses'
        raise NotImplementedError(errormsg)


class Network(Route):
    """
    A class holding a single network of contact edges (connections) between people
    as well as methods for updating these. Networks mediate disease transmission
    between agents in `ss.People`; see `ss.Disease` for how transmission uses
    network edges, and `ss.RandomNet`, `ss.MFNet`, `ss.PrenatalNet` for
    built-in network types.

    The input is typically arrays including: person 1 of the connection, person 2 of
    the connection, the weight of the connection, the duration and start/end times of
    the connection.

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

        # Initialize the keys of the network
        self.edges = sc.objdict()
        self.participant = ss.BoolArr('participant')

        # Standard module arguments are parameters; the rest are edges
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
        for key, dtype in self.meta.items():
            if key not in self.edges:
                self.edges[key] = np.empty((0,), dtype=dtype)

        # Check that p1 and p2 are ss.uids()
        self.validate_uids()
        return

    def init_post(self, add_pairs=True):
        super().init_post()
        if add_pairs: self.add_pairs()
        return

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

    def __getitem__(self, key):
        """ Allow edge columns to be accessed like a dict, e.g. `network['p1']`, as for Covasim v3 layers """
        if isinstance(key, str) and key in self.edges:
            return self.edges[key]
        return super().__getitem__(key)

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
            try:
                new_arr = edges[key]
            except KeyError:
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
            sim = ss.Sim(n_agents=100, networks='mf').init()
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
        p1 = self.edges.p1
        p2 = self.edges.p2

        # Make a lookup table of the UIDs to remove (much faster than np.isin() for many edges)
        n = max(np.max(p1, initial=-1), np.max(p2, initial=-1), np.max(uids, initial=-1)) + 1
        is_removed = np.zeros(n, dtype=bool)
        is_removed[uids] = True

        # Remove edges where either partner is being removed
        remove = is_removed[p1] | is_removed[p2]
        self.remove_edges(remove)
        return

    def net_beta(self, disease_beta=None, inds=None, disease=None):
        """ Calculate the beta for the given disease and network """
        if inds is None: inds = Ellipsis
        return self.edges.beta[inds] * disease_beta # Beta should already include dt if desired


class DynamicNetwork(Network):
    """ A network where partnerships update dynamically """
    def __init__(self, pars=None, edges=None, **kwargs):
        super().__init__(pars=pars, edges=edges, **kwargs)
        self.meta.dur = ss_float # Add duration to the meta keys for dynamic networks
        return

    def step(self):
        """ Remove expired partnerships and add new ones """
        self.end_pairs()
        self.add_pairs()
        return

    def end_pairs(self):
        """ Remove partnerships whose duration has expired or whose members have died. """
        people = self.sim.people
        self.edges.dur = self.edges.dur - 1 # dur is stored in units of self.t.dt (timesteps), so decrement by 1 per step

        # Non-alive agents are removed
        active = (self.edges.dur > 0) & people.alive[self.edges.p1] & people.alive[self.edges.p2]
        for k in self.meta_keys():
            self.edges[k] = self.edges[k][active]
        return len(active)


def check_duration(net, pars, kwargs):
    """ The duration of a partnership is dur=, as for every other network; Starsim 3's duration= is an error that says so """
    if 'duration' in kwargs or 'duration' in (pars or {}):
        errormsg = f'duration= of ss.{type(net).__name__}() is now dur=, as for every other network, e.g. ss.{type(net).__name__}(dur=ss.lognorm_ex(mean=ss.years(2), std=ss.years(1)))'
        raise TypeError(errormsg)
    return


class SexualNetwork(DynamicNetwork):
    """ Base class for all sexual networks """
    def __init__(self, pars=None, edges=None, **kwargs):
        super().__init__(pars=pars, edges=edges, **kwargs)
        self.meta.acts = ss_int # Add acts to the meta keys for sexual networks
        self.debut = ss.FloatArr('debut', default=0)
        return

    def active(self, people):
        """ Return boolean array of agents who are alive, participating, and past sexual debut. """
        valid_age = people.age > self.debut
        active = self.participant & valid_age & people.alive
        return active

    def available(self, people, sex):
        """ Return UIDs of active agents of the given sex who are not currently partnered. """
        # Could modify this to account for concurrency
        # This property could also be overwritten by a NetworkConnector
        # which could incorporate information about membership in other
        # contact networks
        available = people[sex] & self.active(people)
        available[self.edges.p1] = False
        available[self.edges.p2] = False
        return available.uids

    def net_beta(self, disease_beta=None, inds=None, disease=None):
        """ Compute per-edge transmission probability accounting for number of sex acts. """
        if inds is None: inds = Ellipsis
        return self.edges.beta[inds] * (1 - (1 - disease_beta) ** (self.edges.acts[inds]))


# %% Specific instances of networks



class StaticNet(Network):
    """
    A network class of static partnerships converted from a networkx graph. There's no formation of new partnerships
    and initialized partnerships only end when one of the partners dies. The networkx graph can be created outside Starsim
    if population size is known. Or the graph can be created by passing a networkx generator function to Starsim.

    If "seed=True" is passed as a keyword argument or a parameter in pars, it is replaced with the built-in RNG.
    The parameter "n" is supplied automatically to be equal to n_agents.

    Examples:
        ```python
        # Generate a networkx graph and pass to Starsim
        import networkx as nx
        import starsim as ss
        g = nx.scale_free_graph(n=10000)
        ss.StaticNet(graph=g)

        # Pass a networkx graph generator to Starsim
        ss.StaticNet(graph=nx.erdos_renyi_graph, p=0.0001, seed=True)
        ```
    """

    def __init__(self, graph=None, pars=None, **kwargs):
        super().__init__()
        self.nx_graph = graph # Not self.graph, which is the graph of states and transitions of a declared model
        self.define_pars(seed=True, p=None, n_contacts=10)
        self.graph_kwargs = {k:kwargs.pop(k) for k in list(kwargs) if k not in self.pars and k not in ss.modules.module_args} # Other arguments are passed to the graph generator
        self.update_pars(pars, **kwargs)
        self.dist = ss.Dist(name='StaticNet')
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.n_agents = sim.pars.n_agents
        if self.nx_graph is None:
            import networkx as nx # Lazy import since slow
            self.nx_graph = nx.fast_gnp_random_graph # Fast random (Erdos-Renyi) graph creator
        n_contacts = self.pars.pop('n_contacts') # Remove from pars dict, but use only if p is not supplied
        if self.pars.p is None: # Convert from n_contacts to probability
            self.pars.p = n_contacts/self.n_agents
        return

    def init_post(self):
        super().init_post()
        if 'seed' in self.pars and self.pars.seed is True:
            self.pars.seed = self.dist.rng
        if callable(self.nx_graph):
            args = inspect.signature(self.nx_graph).parameters
            kw = {k:v for k,v in self.pars.items() if k in args} # Only pass p and seed if the generator accepts them
            try:
                self.nx_graph = self.nx_graph(n=self.n_agents, **kw, **self.graph_kwargs)
            except TypeError as e:
                ss.warn(f"{str(e)}: networkx {self.nx_graph.name} not supported. Try using ss.StaticNet() instead.")
                raise e
        self.validate_pop(self.n_agents)
        self.get_edges()
        return

    def validate_pop(self, n_agents):
        n_nodes = self.nx_graph.number_of_nodes()
        if n_nodes > n_agents:
            errmsg = (f"Please ensure the number of nodes in graph ({n_nodes}) is less than "
                      f"or equal to (<=) the agent population size ({n_agents}).")
            raise ValueError(errmsg)

        if not self.nx_graph.number_of_edges():
            errmsg = f"The nx generator {self.nx_graph.name} produced a graph with no edges"
            raise ValueError(errmsg)

    def get_edges(self):
        p1s = []
        p2s = []
        for edge in self.nx_graph.edges():
            p1, p2 = edge
            p1s.append(p1)
            p2s.append(p2)
        edges = dict(p1=p1s, p2=p2s, beta=np.ones_like(p1s))
        self.append(edges)
        return

    def step(self):
        pass


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

    None of these is random-number safe; see `ssl.RandomSafeNet` (`import starsim.library as ssl`)
    for a CRN-safe (but slower) version.

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
        if 'uniform_targets' in kwargs: # Starsim 4 prereleases
            errormsg = "uniform_targets=True of ss.RandomNet() is now method='uniform'"
            raise TypeError(errormsg)
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
        source = np.repeat(inds, n_contacts)
        n = len(source)
        if n and self.pars.method == 'uniform':
            idx = self.dist.rng.integers(0, len(inds), n) # A random target agent for each source stub
            target = np.take(np.asarray(inds), idx).view(ss.uids)
        elif n:
            idx = self.dist.rng.integers(0, n, n) # A random target stub for each source stub; sampling stubs rather than agents keeps selection probability proportional to n_contacts
            nc = np.asarray(n_contacts)
            nc0 = int(nc.flat[0])
            # Fast path: if every agent has the same number of stubs, then stub j belongs to agent j//nc0,
            # so the (identical) targets can be gathered from the much smaller, cache-resident agent array
            if nc.size == 1 or (nc == nc0).all():
                np.floor_divide(idx, nc0, out=idx)
                target = np.take(np.asarray(inds), idx).view(ss.uids)
            else:
                target = np.take(np.asarray(source), idx).view(ss.uids) # np.take is slightly faster than source[idx] since it skips the ss.uids subclass machinery
        else:
            target = source
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

    def step(self):
        """ If dynamic, replace expired edges; otherwise, do nothing """
        if self.pars.dynamic:
            super().step()
        return

    def add_pairs(self):
        """ Generate edges """
        p = self.pars
        people = self.sim.people
        uids = self.eligible(p.age_range).uids
        if isinstance(p.n_contacts, ss.Dist):
            n_conn = p.n_contacts.rvs(uids)
        else:
            n_conn = np.full(len(people), p.n_contacts)

        # See how many new edges we need
        orig = n_conn.sum()
        target = orig / 2 # Divide by 2 since bi-directional
        current = len(self)
        needed = target - current
        if needed > 0:
            n_conn = n_conn*needed/orig
            n_conn = p.n_contacts.randround(n_conn) # CRN-safe stochastic integer rounding

            # Get the new edges -- the key step
            p1, p2 = self.pair(uids, n_conn)
            n = len(p1)
            beta = np.full(n, self.pars.beta, dtype=ss_float)

            if isinstance(p.dur, ss.Dist):
                dur = p.dur.rvs(p1)
            elif p.dur == 0:
                dur = np.zeros(n)
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
    Targets for the number of partners people have at a time, for `ss.PartnerNet` and `ss.MFNet`

    The target is a distribution of the number of partners (`degree_dist()`). Partnerships
    end at random, and people look for new ones with a probability that depends on how many
    they already have (`p_seek`), chosen so that the target is the stable state.
    """
    @staticmethod
    def degree_dist(n_contacts, concurrency=None, max_contacts=None):
        """ The target share of people with 0, 1, 2, ... partners at a time """
        if isinstance(n_contacts, ss.by):
            errormsg = f'n_contacts={n_contacts} gives each level its own number of partners, which ss.PartnerNet takes; here give one number or distribution'
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

    def init_seek(self, mean_dur):
        """ The probability that someone with each number of partners looks for another on a step, given how many steps partnerships last on average """
        k = np.arange(len(self.dist_degree))
        p_end = 1/mean_dur
        ratio = np.divide(self.dist_degree[1:], self.dist_degree[:-1], out=np.zeros(len(k) - 1), where=self.dist_degree[:-1] > 0)
        self.p_seek = np.minimum(np.append((k[:-1] + 1)*p_end*ratio, 0), 1) # Chosen so the target is the stationary distribution: those with k partners who gain one balance those with k+1 who lose one
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

    def define_degree_results(self):
        self.define_results(
            ss.Result('mean_degree', label='Mean number of partners', auto_plot=False),
            ss.Result('concurrency', label='Share with two or more partners', auto_plot=False),
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
        across runs; a target that is missed by more than the spread is an `ss.TargetStatsNotRecovered` warning.

        Args:
            n_runs (int): the number of runs
            output (bool): return the table instead of printing it
        """
        if not hasattr(self.sim, 'inputs'):
            errormsg = f'net.check() reruns the sim, which can only be rebuilt from the inputs it was made with; the network\'s results (mean_degree, concurrency) show how close it is to the targets'
            raise ValueError(errormsg)
        rows = []
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
                uids = run.people.alive.uids
                level = np.asarray(net.levels(net.by_level))[uids].astype(int)
                degree = net.degrees(uids)
                for lev in range(len(net.dists_degree)):
                    row[f'mean_degree[{lev}]'] = degree[level == lev].mean() if np.any(level == lev) else np.nan
            if self.pars.get('match') is not None:
                row['match'] = res.match[half:].mean()
            rows.append(row)
        df = sc.dataframe(rows)
        k = np.arange(len(self.dist_degree))
        targets = dict(mean_degree=self.dist_degree @ k, concurrency=self.dist_degree[2:].sum(), max_contacts=self.pars.get('max_contacts'), dur=getattr(self, 'mean_dur', None))
        if getattr(self, 'by_level', None) is not None:
            targets.update({f'mean_degree[{lev}]':d @ np.arange(len(d)) for lev,d in enumerate(self.dists_degree)})
        if self.pars.get('match') is not None:
            targets['match'] = list(self.pars.match.values())[0]
        table = sc.dataframe(dict(target=[targets[c] for c in df.columns], realized=df.mean().values, spread=df.std().values), index=df.columns)
        for name,row in table.iterrows():
            if row.target is None:
                continue
            missed = row.realized > row.target if name == 'max_contacts' else abs(row.realized - row.target) > max(2*row.spread, 0.02*abs(row.target))
            if missed:
                ss.warn(f'the network "{self.name}" has {name} {row.realized:.3g}, against a target of {row.target:.3g}', ss.TargetStatsNotRecovered)
        if output:
            return table
        print(table)
        return


class PartnerNet(DegreeTargets, DynamicNetwork):
    """
    Partnerships that form and dissolve, with a given number of partners at a time

    The number of partners people have at any moment (`n_contacts`) is a target
    for the whole population, not a fixed number for each person: partnerships end
    at random, and people look for new ones at a rate that depends on how many
    they already have, chosen so that the distribution of the number of partners
    stays at the target. Everyone moves through it over time, as in EpiModel's
    networks fitted to target statistics. The network starts at the target.

    Args:
        n_contacts (float/`ss.Dist`/`ss.by`): the number of partners a person has at a time: a distribution over whole numbers (e.g. `ss.poisson(0.7)`), or a number, which is the mean of a Poisson; or a number for each level of a property of people (`ss.by(risk=[0.4, 1.6])`), as EpiModel's nodefactor
        dur (float/`ss.dur`/`ss.Dist`): how long partnerships last: a mean (they end at a constant rate), or a distribution; a bare number is in the sim's unit
        concurrency (float): if given, the share of people with two or more partners at once; the distribution of `n_contacts` is adjusted to give it, keeping its mean
        max_contacts (int): if given, nobody has more partners than this at once
        match (dict): if given, the share of partnerships between people at the same level of a property of people (a stratification, or an attribute such as `'female'`), e.g. `dict(risk=0.8)`, as EpiModel's nodematch
        acts (float/`ss.Dist`/rate): if given, the number of acts in each partnership in a step (a number, a distribution drawn for each partnership, or a rate such as `ss.perweek(2)`); a disease's beta is then per act, so a partnership transmits with probability 1 − (1 − beta)^acts, as EpiModel's act rate
        beta (float): the default beta for each partnership

    The realized network is recorded in the results (`mean_degree`, `concurrency`, and
    `match`), and `net.check()` compares it with the targets.

    Examples:
        ```python
        # EpiModel's introductory network: 175 partnerships among 500 people, 110 people with two or more partners, nobody with four or more, and partnerships lasting 50 steps
        net = ss.PartnerNet(n_contacts=0.7, concurrency=0.22, max_contacts=3, dur=50)

        # Most partnerships within a risk group
        net = ss.PartnerNet(n_contacts=0.7, dur=50, match=dict(risk=0.8))

        # And four times as many partners in the high-risk group (with sim.stratify(risk=['low', 'high']))
        net = ss.PartnerNet(n_contacts=ss.by(risk=[0.4, 1.6]), dur=50, match=dict(risk=0.8))
        ```
    """
    def __init__(self, pars=None, n_contacts=_, dur=_, concurrency=_, max_contacts=_, match=_, acts=_, beta=_, **kwargs):
        super().__init__()
        self.define_pars(
            n_contacts = 1.0,
            dur = 10,
            concurrency = None,
            max_contacts = None,
            match = None,
            acts = None,
            beta = 1.0,
        )
        self.update_pars(pars, **kwargs)
        if self.pars.acts is not None:
            self.meta.acts = ss_float # Acts in each partnership in a step
            if 'acts' not in self.edges:
                self.edges.acts = np.array([], dtype=ss_float)
        self.rng_degree = ss.random() # The number of partners each person starts with
        self.rng_order  = ss.random() # The order in which people are paired
        self.rng_seek   = ss.random() # Whether each person looks for a partner on this step
        self.rng_end    = ss.random() # When each partnership ends, if it ends at a constant rate
        self.rng_stubs  = ss.random() # For shuffling the partners people start with
        self.rng_match  = ss.random() # Whether each person looks for a partner at their own level, with match=
        self.waiting = None # Someone who looked for a partner on the last step and was left over
        self.departures = 0.0 # The share of partnerships ended each step by a partner leaving (dying), estimated as the run goes
        self.lost = 0 # Partnerships ended that way on this step
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        p = self.pars
        self.by_level = None # With n_contacts=ss.by(...): the property of people whose levels have degrees of their own, and the distribution and seeking probabilities of each
        if isinstance(p.n_contacts, ss.by):
            if len(p.n_contacts.dims) != 1 or p.n_contacts.dim is None:
                errormsg = f'n_contacts= of "{self.name}" takes a number for each level of one property of people, e.g. ss.by(risk=[0.4, 1.6]), not {p.n_contacts}'
                raise ValueError(errormsg)
            self.by_level = p.n_contacts.dim
            self.dists_degree = [self.degree_dist(v, p.concurrency, p.max_contacts) for v in sc.tolist(p.n_contacts.values)]
            size = max(len(d) for d in self.dists_degree)
            self.dists_degree = [np.pad(d, (0, size - len(d))) for d in self.dists_degree]
            self.dist_degree = np.mean(self.dists_degree, axis=0) # Replaced by the population's mix of levels at init_post()
        else:
            self.dist_degree = self.degree_dist(p.n_contacts, p.concurrency, p.max_contacts)
        if isinstance(p.dur, ss.Dist):
            self.mean_dur = None
        else:
            dur = ss.solvers.convert(p.dur, sim.unit_base) if isinstance(p.dur, ss.TimePar) else p.dur
            self.mean_dur = dur/self.t.dt.to(sim.unit_base).value # In steps
            if self.mean_dur < 1:
                errormsg = f'Partnerships of "{self.name}" last {self.mean_dur:n} steps on average; they need to last at least one'
                raise ValueError(errormsg)
        if p.match is not None and len(p.match) != 1:
            errormsg = f'match= of "{self.name}" takes one property of people and the share of partnerships at the same level of it, e.g. dict(risk=0.8), not {p.match}'
            raise ValueError(errormsg)
        mean_dur = self.mean_dur if self.mean_dur is not None else self.dwell_mean()
        if self.by_level is not None: # Seeking probabilities for each level
            self.p_seeks = []
            for d in self.dists_degree:
                self.dist_degree = d
                self.init_seek(mean_dur)
                self.p_seeks.append(self.p_seek)
            self.p_seeks = np.array(self.p_seeks)
            self.dist_degree = np.mean(self.dists_degree, axis=0)
        self.init_seek(mean_dur)
        if p.acts is not None:
            if isinstance(p.acts, ss.TimePar):
                self.acts_per_step = ss.solvers.convert(p.acts, sim.unit_base)*self.t.dt.to(sim.unit_base).value
            elif not isinstance(p.acts, ss.Dist):
                self.acts_per_step = float(p.acts)
        return

    def net_beta(self, disease_beta=None, inds=None, disease=None):
        """ With acts=: the probability that a partnership transmits in a step, from beta per act """
        if self.pars.acts is None:
            return super().net_beta(disease_beta=disease_beta, inds=inds, disease=disease)
        if inds is None: inds = Ellipsis
        return self.edges.beta[inds]*(1 - (1 - disease_beta)**self.edges.acts[inds])

    def dwell_mean(self):
        """ The mean of a distribution of durations, in steps """
        return float(np.mean(sc.dcp(self.pars.dur).init(force=True).rvs(100_000)))

    def init_results(self):
        super().init_results()
        self.define_degree_results()
        self.define_results(ss.Result('new_partnerships', label='Partnerships formed', auto_plot=False))
        if self.pars.match is not None:
            self.define_results(ss.Result('match', label='Share of partnerships at the same level', auto_plot=False))
        return

    def levels(self, name=None):
        """ The level of every person in a property of people (by default, the one match= names), as an array indexed by UID """
        sim = self.sim
        name = list(self.pars.match.keys())[0] if name is None else name
        names = [strat.name for strat in sim.groups.strats]
        if name in names: # A stratification
            return sim.groups.level_raw(names.index(name))
        return np.asarray(sim.people[name].raw)

    def match_pairs(self, people, rvs):
        """
        Pair people who are looking for partners, who are in a random order: in turn, or, with `match=`, some within
        their own level, so that the share of partnerships at the same level is the target

        Args:
            people (uids): the people, once for each partner they look for
            rvs (array): a random number for each, which says whether they look within their level

        Returns the two lists of partners, and anyone left over.
        """
        if self.pars.match is None:
            n = len(people)//2
            return people[:n], people[n:2*n], people[2*n:]
        share = list(self.pars.match.values())[0]
        level = self.levels()[people]
        counts = np.unique(level, return_counts=True)[1]
        by_chance = np.sum((counts/max(len(people), 1))**2) # The share at the same level if they were paired at random
        within = np.clip((share - by_chance)/(1 - by_chance), 0, 1) if by_chance < 1 else 1.0
        inside = rvs < within
        p1, p2, rest = [], [], [people[~inside]]
        for lev in np.unique(level[inside]):
            group = people[inside & (level == lev)]
            n = len(group)//2
            p1.append(group[:n])
            p2.append(group[n:2*n])
            rest.append(group[2*n:]) # Anyone left over is paired with the others
        rest = np.concatenate(rest)
        n = len(rest)//2
        p1 = np.concatenate(p1 + [rest[:n]]).view(ss.uids)
        p2 = np.concatenate(p2 + [rest[n:2*n]]).view(ss.uids)
        return p1, p2, rest[2*n:].view(ss.uids)

    def end_times(self, p1):
        """ How many steps each new partnership lasts """
        if self.mean_dur is None:
            return self.pars.dur.rvs(p1)
        u = self.rng_end.rvs(p1)
        p_end = max(1/self.mean_dur - self.departures, 0.1/self.mean_dur) # Partnerships also end when a partner leaves, so they are dissolved less often, to last as long as given on average (as EpiModel's d.rate correction does)
        return np.maximum(np.ceil(np.log1p(-u)/np.log1p(-p_end)), 1) # Geometric, so partnerships end at a constant rate

    def pair(self, p1, p2):
        """ Add partnerships between two lists of people, leaving out pairs who are already partners """
        if len(self):
            existing = set(zip(np.minimum(self.edges.p1, self.edges.p2).tolist(), np.maximum(self.edges.p1, self.edges.p2).tolist()))
            new = zip(np.minimum(p1, p2).tolist(), np.maximum(p1, p2).tolist())
            keep = np.array([pair not in existing for pair in new], dtype=bool)
            p1, p2 = p1[keep], p2[keep]
        keep = p1 != p2
        p1, p2 = p1[keep], p2[keep]
        acts = {} if self.pars.acts is None else dict(acts=self.pars.acts.rvs(p1).astype(ss_float) if isinstance(self.pars.acts, ss.Dist) else np.full(len(p1), self.acts_per_step, dtype=ss_float))
        self.append(p1=p1, p2=p2, beta=np.full(len(p1), self.pars.beta, dtype=ss_float), dur=self.end_times(p1), **acts)
        return

    def init_post(self):
        """ Start at the target: give each person a number of partners, and pair them at random (with match=, some within their level) """
        super().init_post(add_pairs=False)
        uids = self.sim.people.alive.uids
        if self.by_level is not None: # Each level from its own distribution
            level = np.asarray(self.levels(self.by_level))[uids].astype(int)
            cdfs = np.cumsum(self.dists_degree, axis=1)
            u = self.rng_degree.rvs(uids)
            degrees = np.array([np.searchsorted(cdfs[lev], ui) for lev,ui in zip(level, u)], dtype=int)
            degrees = np.minimum(degrees, cdfs.shape[1] - 1)
            shares = np.bincount(level, minlength=len(self.dists_degree))/max(len(level), 1)
            self.dist_degree = shares @ self.dists_degree # The population's target, for the results and check()
        else:
            degrees = self.start_degrees(uids, self.rng_degree)
        stubs = np.repeat(uids, degrees)
        fisher_yates_shuffle(stubs, self.rng_stubs.rng.random(len(stubs)))
        self.rng_stubs.jump() # Reset the RNG manually, as in RandomNet.shuffle_edges()
        rvs = None
        if self.pars.match is not None: # By position, since each person appears once for each partner
            rvs = self.rng_stubs.rng.random(len(stubs))
            self.rng_stubs.jump()
        p1, p2, left = self.match_pairs(stubs.view(ss.uids), rvs)
        self.pair(p1, p2)
        return

    def remove_uids(self, uids):
        """ Remove the partnerships of people who leave, counting them (see `end_times()`) """
        n = len(self)
        super().remove_uids(uids)
        self.lost += n - len(self)
        return

    def add_pairs(self):
        """ People look for partners at a rate that depends on how many they have, and those looking are paired at random """
        if len(self) or self.lost: # The share of partnerships ended by a partner leaving, smoothed over about 20 steps
            self.departures += 0.05*(self.lost/max(len(self) + self.lost, 1) - self.departures)
            self.lost = 0
        uids = self.sim.people.alive.uids
        degree = np.minimum(self.degrees(uids), len(self.p_seek) - 1)
        if self.by_level is not None:
            prob = self.p_seeks[np.asarray(self.levels(self.by_level))[uids].astype(int), degree]
        else:
            prob = self.p_seek[degree]
        seek = self.rng_seek.rvs(uids) < prob
        if self.waiting is not None: # Someone left over last step looks again
            seek |= (uids == self.waiting) & (prob > 0)
        seekers = uids[seek]
        seekers = seekers[np.argsort(self.rng_order.rvs(seekers), kind='stable')]
        p1, p2, left = self.match_pairs(seekers, self.rng_match.rvs(seekers) if self.pars.match is not None else None)
        self.waiting = left[-1] if len(left) else None
        n = len(self)
        if len(p1):
            self.pair(p1, p2)
        self.formed = len(self) - n # For the duration in check()
        return

    def update_results(self):
        super().update_results()
        self.update_degree_results(self.sim.people.alive.uids)
        self.results.new_partnerships[self.ti] = getattr(self, 'formed', 0)
        if self.pars.match is not None and len(self):
            level = self.levels()
            self.results.match[self.ti] = np.mean(level[self.edges.p1] == level[self.edges.p2])
        return

class MFNet(DegreeTargets, SexualNetwork):
    """
    This network is built by **randomly pairing** males and female with variable
    relationship durations.

    By default, people who have no partner are paired, so nobody has more than one at
    a time. With `n_contacts`, the number of partners people have at a time is a target
    instead, as for `ss.PartnerNet`: partnerships end when their duration is up, and
    people look for new ones at a rate that depends on how many they have, so the
    distribution of the number of partners (among people who take part, after their
    debut) stays at the target. The network then starts at the target, and its results
    and `net.check()` say how close it is.

    Args:
        dur (`ss.Dist`): the duration of each partnership; can vary by age, year, and individual pair. Set scale=exp(mu) and s=sigma where mu,sigma are of the underlying normal distribution.
        debut (`ss.Dist`): Age of debut can vary by using callable parameter values
        acts (`ss.Dist`): Number of acts per year
        participation (`ss.Dist`): Probability of participating in this network - can vary by individual properties (age, sex, ...) using callable parameter values
        rel_part_rates (float): Relative participation in the network, i.e. a multiplier (from 0 to 1) on the probability of participating
        n_contacts (float/`ss.Dist`): if given, the number of partners people have at a time: a distribution over whole numbers, or a number, which is the mean of a Poisson
        concurrency (float): with `n_contacts`, the share of people with two or more partners at once
        max_contacts (int): with `n_contacts`, the most partners anyone has at once

    Examples:
        ```python
        net = ss.MFNet(n_contacts=ss.poisson(0.7), dur=ss.lognorm_ex(mean=ss.years(2), std=ss.years(1)), concurrency=0.1)
        ```
    """
    def __init__(self, pars=None, dur=_, debut=_, acts=_, participation=_, rel_part_rates=_, n_contacts=_, concurrency=_, max_contacts=_, **kwargs):
        super().__init__()
        check_duration(self, pars, kwargs)
        self.define_pars(
            dur = ss.lognorm_ex(mean=ss.years(15), std=ss.years(1)),  # Can vary by age, year, and individual pair. Set scale=exp(mu) and s=sigma where mu,sigma are of the underlying normal distribution.
            debut = ss.normal(loc=16),  # Age of debut can vary by using callable parameter values
            acts = ss.poisson(lam=ss.freqperyear(80)),
            participation = ss.bernoulli(p=0.9),  # Probability of participating in this network - can vary by individual properties (age, sex, ...) using callable parameter values
            rel_part_rates = 1.0,
            n_contacts = None,
            concurrency = None,
            max_contacts = None,
        )
        self.update_pars(pars, **kwargs)

        # Finish initialization
        self.dist = ss.choice(name='MFNet', replace=False) # Set the array later
        self.rel_part = ss.bernoulli(p=1.0) # Set p later, in case rel_part_rates is modified
        if self.targeted: # Only then, so the random numbers of a network without targets are unchanged
            self.rng_degree = ss.random() # The number of partners each person starts with
            self.rng_seek   = ss.random() # Whether each person looks for a partner on this step
            self.rng_order  = ss.random() # The order in which people are paired
            self.waiting = ss.uids() # People of one sex who looked for a partner and were left over, who look again
        return

    @property
    def targeted(self):
        """ Whether the number of partners people have at a time is a target (`n_contacts`), rather than one at most """
        return self.pars.n_contacts is not None

    def init_pre(self, sim):
        super().init_pre(sim)
        p = self.pars
        if self.targeted:
            self.dist_degree = self.degree_dist(p.n_contacts, p.concurrency, p.max_contacts)
            mean = ss.transitions.dwell_stats(p.dur, lambda v: ss.solvers.convert(v, sim.unit_base))[0]
            self.init_seek(mean/self.t.dt.to(sim.unit_base).value) # In steps
        elif p.concurrency is not None or p.max_contacts is not None:
            errormsg = f'concurrency= and max_contacts= of "{self.name}" are targets for the number of partners at a time, so need n_contacts= as well'
            raise ValueError(errormsg)
        return

    def init_results(self):
        super().init_results()
        if self.targeted:
            self.define_degree_results()
        return

    def people_seeking(self, people):
        """ The people who take part, after their debut """
        return self.active(people).uids

    def init_post(self):
        self.set_network_states()
        if not self.targeted:
            super().init_post()
            return
        super().init_post(add_pairs=False) # Start at the target: give each person who takes part a number of partners, and pair men with women at random
        people = self.sim.people
        uids = self.people_seeking(people)
        stubs = np.repeat(uids, self.start_degrees(uids, self.rng_degree))
        rng = self.rng_order.rng # By position, since each person appears once for each partner
        p1, p2 = self.pair_sexes(stubs, rng.random(len(stubs)))
        n = len(p1)
        dur = self.pars.dur.rvs(3*n) # What is left of partnerships under way: one picked in proportion to how long it lasts, and a uniform share of it
        dur = dur[rng.choice(len(dur), size=n, p=dur/dur.sum())]*rng.random(n) if n else dur[:0]
        self.rng_order.jump()
        self.add(p1, p2, dur)
        return

    def pair_sexes(self, people, order):
        """ Pair the men and the women among some people, in the order of a random number for each, leaving out pairs who are already partners; anyone left over waits """
        people = people[np.argsort(order, kind='stable')]
        female = self.sim.people.female.raw[people]
        men, women = people[~female], people[female]
        n = min(len(men), len(women))
        p1, p2 = men[:n], women[:n]
        self.waiting = ss.uids.concatenate([men[n:], women[n:]]).unique() # Otherwise the surplus of one sex would be lost on every step
        if len(self) and n:
            existing = set(zip(self.edges.p1.tolist(), self.edges.p2.tolist()))
            keep = np.array([pair not in existing for pair in zip(p1.tolist(), p2.tolist())], dtype=bool)
            p1, p2 = p1[keep], p2[keep]
        return p1, p2

    def add(self, p1, p2, dur=None):
        """ Add partnerships, with their durations (drawn if not given) and acts """
        if len(p1):
            dur = self.pars.dur.rvs(len(p1)) if dur is None else dur
            self.append(p1=p1, p2=p2, beta=np.ones(len(p1)), dur=dur, acts=self.pars.acts.rvs(len(p1)))
        return len(p1)

    def set_network_states(self, upper_age=None):
        """ Set network states including age of entry into network and participation rates """
        self.set_debut(upper_age=upper_age)
        self.set_participation(upper_age=upper_age)
        return

    def set_participation(self, upper_age=None):
        """ Set people who will participate in the network at some point """
        people = self.sim.people
        if upper_age is None: uids = people.auids
        else: uids = (people.age < upper_age).uids
        participant = self.pars.participation.rvs(uids)
        if self.pars.rel_part_rates != 1: # Scale the probability of participating
            self.rel_part.set(p=self.pars.rel_part_rates)
            participant &= self.rel_part.rvs(uids)
        self.participant[uids] = participant
        return

    def set_debut(self, upper_age=None):
        """ Set debut age """
        people = self.sim.people
        if upper_age is None: uids = people.auids
        else: uids = (people.age < upper_age).uids
        self.debut[uids] = self.pars.debut.rvs(uids)
        return

    def add_pairs(self):
        """ Form new sexual partnerships between available male and female agents """
        people = self.sim.people
        if self.targeted: # Those looking for a partner on this step, given how many they have
            active = self.people_seeking(people)
            seekers = self.seeking(active, self.rng_seek)
            waiting = self.waiting[np.isin(self.waiting, active)]
            waiting = waiting[self.p_seek[np.minimum(self.degrees(waiting), len(self.p_seek) - 1)] > 0] # Unless they have as many partners as anyone can now
            seekers = ss.uids.concatenate([seekers, waiting]).unique()
            p1, p2 = self.pair_sexes(seekers, self.rng_order.rvs(seekers))
            return self.add(p1, p2)
        available_m = self.available(people, 'male')
        available_f = self.available(people, 'female')

        # random.choice is not common-random-number safe, and therefore we do
        # not try to Stream-ify the following draws at this time.
        if len(available_m) <= len(available_f):
            self.dist.set(a=available_f)
            p1 = available_m
            p2 = self.dist.rvs(n=len(p1)) # TODO: not fully stream safe
        else:
            self.dist.set(a=available_m)
            p2 = available_f
            p1 = self.dist.rvs(n=len(p2))
        self.dist.jump() # TODO: think if there's a better way

        beta = np.ones_like(p1)

        # Figure out durations and acts
        if (len(p1) == len(np.unique(p1))):
            # No duplicates and user has enabled multirng, so use slotting based on p1
            dur_vals = self.pars.dur.rvs(p1)
            act_vals = self.pars.acts.rvs(p1)
        else: # TODO: rethink explanation without multirng
            # If multirng is enabled, we're here because some individuals in p1
            # are starting multiple relationships on this timestep. If using
            # slotted draws, as above, repeated relationships will get the same
            # duration and act rates, which is scientifically undesirable.
            # Instead, we fall back to a not-CRN safe approach:
            dur_vals = self.pars.dur.rvs(len(p1))  # Just use len(p1) to say how many draws are needed
            act_vals = self.pars.acts.rvs(len(p1))

        self.append(p1=p1, p2=p2, beta=beta, dur=dur_vals, acts=act_vals)

        return len(p1)

    def step(self):
        self.end_pairs()
        self.set_network_states(upper_age=self.t.dt.years) # Assign debut age and participation to agents born since the last step, i.e. those younger than one timestep; ages are in years, so the timestep must be too
        self.add_pairs()
        return

    def update_results(self):
        super().update_results()
        if self.targeted:
            self.update_degree_results(self.people_seeking(self.sim.people))
        return


class PrenatalNet(Network):
    """
    Prenatal transmission network

    Although edges change over time, the edges are managed via ss.Pregnancy as pregnancies start and finish.
    The prenatal network therefore is not derived from DynamicNetwork as it does not manage its own updates.
    """

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

    For now the same as `ss.PrenatalNet` (as in Starsim 3), but named `maternalnet`.
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

    Args:
        pars (dict): passed to `ss.DynamicNetwork`
        dur (ss.dur/ss.Dist): edge duration (postpartum period); if None, edges persist until agents die
        **kwargs (dict): merged with `pars`
    """

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
    status from the `Pregnancy` module.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(pars=pars, **kwargs)
        assert self.pars.dur is None, 'BreastfeedingNet does not accept a `dur` parameter as breastfeeding duration is determined by the Pregnancy module'
        return

    def init_pre(self, sim):
        # Connect the pregnancy module to this network; do this here rather than in init_post() since deliveries during the pregnancy burn-in call add_pairs()
        super().init_pre(sim)
        for mod in self.sim.modules:
            if isinstance(mod, ss.Pregnancy):
                self.pregnancy = mod
                break
        else:
            raise RuntimeError('BreastfeedingNet requires a Pregnancy module in the simulation to track breastfeeding status')
        return

    def add_pairs(self, mother_uids=None, newborn_uids=None):
        """
        Add pairs upon delivery if the mother is breastfeeding the newborn - called via Pregnancy.process_delivery()
        """
        if mother_uids is not None:
            mask = self.pregnancy.breastfeeding[mother_uids] & self.pregnancy.breastfed[newborn_uids]
            super().add_pairs(mother_uids[mask], newborn_uids[mask])
        return

    def end_pairs(self):
        """Remove edges for mothers whose breastfeeding status has changed in the Pregnancy module"""
        active = self.pregnancy.breastfeeding[self.p1]
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
        return

    def __len__(self):
        try:    return len(self.pools)
        except: return 0

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

    def init_pre(self, sim):
        super().init_pre(sim)
        p = self.pars
        time_args = {k:p.get(k) for k in ss.Timeline.time_args} # get() allows None

        self.pools = []
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
                    raise TypeError(f'Diseases can be specified as ss.Disease objects or strings, not {type(d)}')
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

        This is called from Infection.infect() together with network transmission.

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


def contacts(matrix=None, bands=None, name=None, source='prem2021', layer='all', **dims):
    """
    Mixing between groups from a contact matrix; returns `ss.MixingPools`

    There are three forms. A country's matrix by name (`ss.contacts('Kenya')`) comes
    from `starsim_data`, by 5-year age group. A matrix with the lower edge of each of its
    own age bands (`ss.contacts(matrix, bands=[0, 15, 65])`) has its own age groups, which
    need not be the model's. A matrix over each of one or more of the model's
    stratifications (`ss.contacts(age=..., location=...)`) uses the model's own levels,
    and the contacts between two groups are the product of the entries for each dimension;
    so `age=` is always a matrix over the model's age groups.

    Args:
        matrix (array/str): a country, by name or code ('Kenya', 'KE'); or `matrix[i,j]`, the number of contacts per unit time that one person in age band `i` has with people in age band `j`
        bands (list/array): the lower edge of each age band of `matrix`, e.g. `[0, 15, 65]`
        name (str): the name of the mixing (default: 'contacts', or the layer, e.g. 'home')
        source (str): for a country, the study: 'prem2021' (the default; bundled, 177 countries), 'prem2017' (152), or 'mistry2021' (36, by single year of age); the last two are downloaded on first use
        layer (str): for a country, the setting: 'all' (the default), 'home', 'school', 'work', or 'community'
        dims (dict): a matrix for each of the model's stratifications, e.g. `age=[[5, 1], [1.5, 2]]` or `location=[[0.8, 0.2], [0.2, 0.8]]`

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        mixing = ss.contacts([[10, 3], [3, 5]], bands=[0, 18])
        sim = ss.Sim(sir, networks=mixing, unit='day', stop=100).run()

        sir.stratify(risk=['low', 'high'])
        mixing = ss.contacts(risk=[[2, 1], [1, 8]])

        home, school = ss.contacts('Kenya', layer='home'), ss.contacts('Kenya', layer='school') # Contacts a day, by 5-year age group
        sim = ss.Sim(sir, networks=[home, school], stratify=dict(age=[0, 5, 18, 65]), unit='day', stop=100)
        ```
    """
    name = name or (layer if layer != 'all' else 'contacts')
    if matrix is not None and 'age' in dims: # Starsim 4 prereleases gave the matrix's own age bands as age=
        errormsg = 'ss.contacts(matrix, age=[...]) is now ss.contacts(matrix, bands=[...]): bands= are the lower edges of the age bands of the matrix given, and age= is a matrix over the model\'s own age groups, e.g. ss.contacts(age=[[5, 1], [1.5, 2]])'
        raise TypeError(errormsg)
    if matrix is None and bands is None and dims: # A matrix for each of the model's own dimensions
        dims = {dim:np.array(mat, dtype=float) for dim,mat in dims.items()}
        for dim,mat in dims.items():
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                errormsg = f'ss.contacts({dim}=...) needs a square matrix, with a row and a column for each level of "{dim}"; got shape {mat.shape}'
                raise ValueError(errormsg)
            check_contacts(mat, dim)
        levels = list(itertools.product(*[range(len(mat)) for mat in dims.values()]))
        groups = {}
        for combo in levels:
            group = LevelGroup(**dict(zip(dims.keys(), combo)))
            groups[group.name] = group
        matrix = functools.reduce(np.kron, dims.values(), np.ones((1, 1))) # The product of the entries for each dimension, for each pair of combinations of levels (in the order of `levels`)
        out = MixingPools(src=groups, dst=sc.dcp(groups), n_contacts=matrix.T, name=name) # ss.MixingPools is indexed [source, destination]; kept as numbers, since a matrix with many levels would be a million objects to make and copy
        out.constant = True # Everyone in a group has the same number of contacts
        out.dims = dims # So a compartmental run can use the matrices directly
        return out
    kwargs = dict(dims)
    ref = None
    if isinstance(matrix, str): # A country's matrix, from starsim_data
        matrix, bands, ref = ss.named.contact_matrix(matrix, source=source, layer=layer)
    elif matrix is None:
        errormsg = 'ss.contacts() needs a matrix: a country (ss.contacts("Kenya")), a matrix with its age groups (ss.contacts(matrix, bands=[...])), or a matrix for each stratification (ss.contacts(age=...))'
        raise ValueError(errormsg)
    matrix = np.array(matrix, dtype=float)
    if bands is None or len(bands) != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        errormsg = f'ss.contacts() needs a square matrix and the lower edge of each of its age bands, e.g. ss.contacts(matrix, bands=[0, 15, 65]); got a matrix of shape {matrix.shape} and bands={bands}'
        raise ValueError(errormsg)
    check_contacts(matrix, 'age')
    if np.any(np.diff(np.array(bands, dtype=float)) <= 0):
        errormsg = f'The age bands of ss.contacts() are given by their lower edges, which need to increase, e.g. bands=[0, 15, 65]; got bands={list(bands)}'
        raise ValueError(errormsg)
    highs = list(bands[1:]) + [None]
    groups = {(f'{lo}-{hi}' if hi is not None else f'{lo}+'): AgeGroup(lo, hi) for lo,hi in zip(bands, highs)}
    kwargs.setdefault('name', name)
    out = MixingPools(src=groups, dst=sc.dcp(groups), n_contacts=matrix.T, **kwargs) # ss.MixingPools is indexed [source, destination]
    out.constant = True # Everyone in a group has the same number of contacts
    out.ref = ref # Where a named matrix came from, for the model spec
    out.bands = None if ref is not None else np.asarray(bands).tolist() # A matrix given with its own age bands, for the model spec (as given, since they name the groups)
    return out


#%% Places and the movement between them

class geo:
    """
    A geography: named places with populations and positions, used as the levels of a spatial stratification

    A country's administrative units come from `starsim_data` by name (`ss.geo('Kenya', level=1)`):
    their names, populations, centroids, areas, p-codes, and which share a border. The first
    two levels of every country are bundled, so this needs no download.

    Args:
        data (dataframe/str/list): a country (or a list of them) by name or code, e.g. 'Kenya' or 'KE'; or a table (or the name of a file holding one) with a column `name`, and optionally `pop`, and `x` and `y` or `lat` and `lon`
        level (int/str): for a country, the administrative level: 1 (the default) for its first-level units (counties, provinces, states), 2 for the next (fetched), 0 for the country itself
        year (int): for a country, the year of the populations (default: those bundled, for 2020; another year is fetched)
        places (list): for a country, only these units, by name, `dot_name` ('Kenya:Nairobi'), or p-code
        names (list): the name of each place (instead of `data`)
        pop (list): the population of each place; sets the share of people who start in each (for a country, in place of the bundled ones)
        x (list): a position for each place, in any unit of distance
        y (list): the other coordinate
        lat (list): latitudes in degrees (instead of `x` and `y`); distances are then great-circle distances in km
        lon (list): longitudes in degrees
        distances (array): the distance between each pair of places (instead of positions)
        adjacency (array): whether each pair of places shares a border

    Examples:
        ```python
        kenya = ss.geo('Kenya', level=1) # Its 47 counties
        towns = ss.geo(names=['north', 'south', 'east'], pop=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        sir.stratify(patch=towns, mobility=ss.gravity(k=0.01))
        ```
    """
    def __init__(self, data=None, names=None, pop=None, x=None, y=None, lat=None, lon=None, distances=None, adjacency=None, level=1, year=None, places=None):
        self.ref = None # Where a named geography came from, for the model spec
        self.area = None
        self.codes = None
        named = isinstance(data, str) and not data.endswith(('.csv', '.xlsx')) or (isinstance(data, (list, tuple)) and all(isinstance(d, str) for d in data))
        if named: # A country's administrative units, from starsim_data
            found = ss.named.geography(data, level=level, year=year, places=places)
            names, lat, lon, adjacency = [found[key] for key in ['names', 'lat', 'lon', 'adjacency']]
            pop = found['pop'] if pop is None else pop # Populations given by hand win
            self.area, self.codes, self.ref = found['area'], found['codes'], found['ref']
            data = None
        elif isinstance(data, str):
            data = sc.dataframe.read_csv(data) if data.endswith('.csv') else sc.dataframe.read_excel(data)
        if data is not None:
            cols = {key:data[key].tolist() for key in ['name', 'pop', 'x', 'y', 'lat', 'lon'] if key in data.columns}
            names, pop, x, y, lat, lon = [cols.get(key) for key in ['name', 'pop', 'x', 'y', 'lat', 'lon']]
        if names is None:
            errormsg = 'ss.geo() needs the names of the places: names=[...], or a table with a column called "name"'
            raise ValueError(errormsg)
        self.names = list(names)
        self.pop = None if pop is None else np.array(pop, dtype=float)
        self.lat = None if lat is None else np.array(lat, dtype=float)
        self.lon = None if lon is None else np.array(lon, dtype=float)
        self.adjacency = None if adjacency is None else np.array(adjacency, dtype=bool)
        self.distances = None
        if distances is not None:
            self.distances = np.array(distances, dtype=float)
        elif x is not None and y is not None:
            x, y = np.array(x, dtype=float), np.array(y, dtype=float)
            self.distances = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        elif lat is not None and lon is not None: # Great-circle distances, in km
            lat, lon = np.radians(np.array(lat, dtype=float)), np.radians(np.array(lon, dtype=float))
            a = np.sin((lat[:, None] - lat[None, :])/2)**2 + np.cos(lat[:, None])*np.cos(lat[None, :])*np.sin((lon[:, None] - lon[None, :])/2)**2
            self.distances = 2*6371*np.arcsin(np.sqrt(a))
        for key,val in dict(pop=self.pop, distances=self.distances, adjacency=self.adjacency).items():
            if val is not None and len(val) != len(self.names):
                errormsg = f'ss.geo() has {len(self.names)} places but {len(val)} entries in {key}'
                raise ValueError(errormsg)
        return

    def __repr__(self):
        if self.ref is not None:
            return f"geo({self.ref['place']!r}, level={self.ref['level']}: {len(self)} places)"
        return f'geo({sc.strjoin(self.names)})'

    def __len__(self):
        return len(self.names)


class gravity:
    """
    The gravity model of movement: the share of their time that a person from place `i` spends in place `j` (or, for migration,
    the rate at which they move there) is `k × pop[i]^(a−1) × pop[j]^b / distance[i,j]^c`

    Populations and distances come from the geography of the dimension it is used on (`ss.geo`).
    The model gives the shape of the movement, and `away=` its amount: the share of their time
    that people spend away from home, split between the other places by gravity. `k` has units
    of distance^c / pop^(a+b−1), so no default fits both three towns and a country; give it only
    if it has been calibrated. With neither, people spend 10% of their time away.

    Args:
        away (float): for commuting and visiting, the share of their time people spend away from home (default 0.1, unless `k` is given)
        k (float): the overall amount of movement, instead of `away`; for migration, the scale of the rates (required)
        a (float): the exponent of the population of the place people come from
        b (float): the exponent of the population of the place they go to
        c (float): the exponent of the distance between the two

    Examples:
        ```python
        sir.stratify(patch=towns, mobility=ss.gravity(away=0.1, a=1.0, b=1.0, c=2.0))
        sir.stratify(patch=towns, mobility=ss.gravity(k=2e-4)) # A calibrated amount
        ```
    """
    def __init__(self, away=None, k=None, a=1.0, b=1.0, c=2.0):
        if away is not None and k is not None:
            errormsg = 'ss.gravity() takes away= (the share of time spent away) or k= (a calibrated amount of movement), not both'
            raise ValueError(errormsg)
        self.away = away
        self.k = k
        self.a = a
        self.b = b
        self.c = c
        return

    def __repr__(self):
        amount = f'away={self.away}, ' if self.away is not None else f'k={self.k}, ' if self.k is not None else ''
        return f'gravity({amount}a={self.a}, b={self.b}, c={self.c})'

    def matrix(self, strat):
        """ The movement between each pair of levels of a stratification, from its geography """
        place = strat.geo
        if place is None or place.distances is None or place.pop is None:
            raise ss.NoGeography(f'ss.gravity() is used on "{strat.name}", which has no geography to give the populations and distances it needs.\n'
                                 f'     Give the places with ss.geo(names=[...], pop=[...], x=[...], y=[...]), or give the movement itself with ss.Mobility(matrix=...).')
        pop, dist = place.pop, place.distances
        k, a, b, c = [float(v) for v in [sc.ifelse(self.k, 1.0), self.a, self.b, self.c]] # Any can be a fitted ss.par
        with np.errstate(divide='ignore'):
            out = k*pop[:, None]**(a - 1)*pop[None, :]**b/dist**c # Without k, only the shape matters: see Mobility.time_share()
        np.fill_diagonal(out, 0)
        return out


class Mobility:
    """
    Movement between the levels of a spatial stratification

    People either commute (they keep their home, and spend a share of their time in
    other places, where they are exposed and expose others), visit (the same, but
    they are exposed to the residents of the places they go to and do not expose
    anyone there), or migrate (they relocate). Or, with `kind='force'`, nobody moves
    and the force of infection does, as in LASER: with `M[i,k]` the movement from
    `i` to `k` (scaled by `away=`), each place keeps `1 − Σ_k M[i,k]` of its own
    force of infection and receives `M[k,i]` of each other place's.

    A movement register (`data=`) is a table of events on given dates, with columns
    `date`, `from`, `to`, `n`, and optionally `states`. Each row moves `n` people from
    one place to another; a row with no `from` adds them to `to` (births, purchases),
    and one with no `to` removes them from `from` (deaths, sales out of the model).
    The people who move or leave are chosen at random from those in `states` (by
    default, everyone); people who are added go into `states` (by default, the first
    state). Each event happens at the start of the first step that begins on or after
    its date, before that step's transitions, and the events of a step happen in the
    order of the table.

    Args:
        model (gravity): a movement model (instead of `matrix`)
        matrix (array): the movement from each place (rows) to each place: for commuting the share of time spent there, and for migration the rate of moving there (a SciPy sparse array for many places, most pairs of which share no one)
        data (str/df): a movement register: a file name or a dataframe (instead of `model` or `matrix`)
        kind (str): 'commute', 'visit', 'migrate', or 'force'; by default 'migrate' for a register and 'commute' otherwise
        away (float): for commuting and visiting, the share of their time people spend away from home (for 'force', the share of each place's force of infection that leaves it); the movement is rescaled to give it
        scale (str): how `away=` rescales the movement: 'each' (the default) gives every place that share; 'mean' keeps the differences between places and gives them that share on average
        cap (float): if given, the most that any place's share can be; a place over it is scaled down to it

    Examples:
        ```python
        sir.stratify(patch=towns, mobility=ss.Mobility(ss.gravity(), kind='commute', away=0.4))

        # LASER's coupling: the mean share exported over places is 4.46%, and no place exports more than 20%
        sir.stratify(patch=places, mobility=ss.Mobility(ss.gravity(a=1, b=0.25, c=1.56), kind='force', away=0.0446, scale='mean', cap=0.2))
        sir.stratify(patch=['a', 'b'], mobility=ss.Mobility(matrix=[[0, 0.1], [0.2, 0]]))
        sir.stratify(herd=herds, mobility=ss.Mobility(data='movements.csv'))
        ```
    """
    def __init__(self, model=None, matrix=None, kind=None, away=None, scale='each', cap=None, data=None):
        if sum(arg is not None for arg in [model, matrix, data]) != 1:
            errormsg = 'ss.Mobility() takes one of a movement model (e.g. ss.gravity()), matrix=, or data='
            raise ValueError(errormsg)
        if kind is None:
            kind = 'migrate' if data is not None else 'commute'
        if kind not in ['commute', 'visit', 'migrate', 'force']:
            errormsg = f'The kind of ss.Mobility() is "commute", "visit", "migrate", or "force", not "{kind}"'
            raise ValueError(errormsg)
        if data is not None and kind != 'migrate':
            errormsg = f'A movement register moves people from one place to another, so its kind is "migrate", not "{kind}"'
            raise ValueError(errormsg)
        self.guessed = None
        if isinstance(model, gravity): # The amount of movement: away= here, or on the gravity model, or its k
            away = sc.ifelse(away, model.away)
            if kind == 'migrate' and model.k is None:
                errormsg = 'ss.gravity() for migration needs k=, the scale of the rates of moving; the share of time away (away=) is for commuting and visiting'
                raise ValueError(errormsg)
            if away is None and model.k is None:
                away = 0.1
                self.guessed = '10% of time away, since neither away= nor k= was given'
        self.model = model
        self.given = None if matrix is None else spsp.csr_array(matrix, dtype=float) if spsp.issparse(matrix) else np.array(matrix, dtype=float)
        self.data = None if data is None else self.load(data)
        if scale not in ['each', 'mean']:
            errormsg = f'scale= of ss.Mobility() is "each" (every place has the share given by away=) or "mean" (the places have it on average), not "{scale}"'
            raise ValueError(errormsg)
        if (scale == 'mean' and away is None) or (cap is not None and kind == 'migrate'):
            errormsg = 'scale= and cap= of ss.Mobility() rescale the share of time away, so they need away= and are not for migration'
            raise ValueError(errormsg)
        self.kind = kind
        self.away = away
        self.scale = scale
        self.cap = cap
        return

    def fitted(self):
        """ The parameters (`ss.par`) in the movement, by attribute (`away`, `cap`, and the gravity model's `k`, `a`, `b`, `c`), which set the mixing when a sim is built """
        objs = dict(away=self, cap=self) | ({key:self.model for key in ['k', 'a', 'b', 'c']} if isinstance(self.model, gravity) else {})
        return {key:getattr(obj, key) for key,obj in objs.items() if isinstance(getattr(obj, key), ss.par)}

    def __repr__(self):
        if self.model is not None:
            what = self.model
        elif self.data is not None:
            what = f'register of {len(self.data):n} events'
        elif self.given.shape[0] <= 4 and not spsp.issparse(self.given): # A small matrix is shown, as the textbooks write it
            what = 'matrix=[' + ', '.join('[' + ', '.join(f'{v:n}' for v in row) + ']' for row in self.given) + ']'
        else:
            what = f'a {self.given.shape[0]}×{self.given.shape[1]} matrix'
        away = f', away={self.away}' if self.away is not None else ''
        away += f', scale={self.scale!r}' if self.scale != 'each' else ''
        away += f', cap={self.cap}' if self.cap is not None else ''
        return f'Mobility({what}, kind={self.kind!r}{away})'

    @staticmethod
    def blank(v):
        """ Whether an entry of a register is empty """
        return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip())

    @staticmethod
    def by_value(vals, func, missing):
        """
        A function of each entry of a column of a register, worked out once for each distinct value, since a register can have millions of rows

        Returns the results for the distinct values, then `missing` (for None and NaN), and the index of each entry's result.
        """
        codes, uniq = pd.factorize(np.asarray(vals, dtype=object)) # None and NaN are -1, which is the last result
        return [func(v) for v in uniq] + [missing], codes

    @staticmethod
    def load(data):
        """ Read a movement register, with a blank `from` or `to` (or a blank `states`) as None """
        df = sc.dataframe(pd.read_csv(data) if isinstance(data, str) else data)
        missing = [col for col in ['date', 'from', 'to', 'n'] if col not in df.columns]
        if missing:
            errormsg = f'A movement register needs the columns date, from, to, and n (and optionally states), but {sc.strjoin(missing)} {"is" if len(missing) == 1 else "are"} missing; its columns are {list(df.columns)}'
            raise KeyError(errormsg)
        out = sc.dataframe(date=df['date'].values, n=np.asarray(df['n'], dtype=float))
        empty = sc.objdict() # Whether each entry of each column is blank
        for col in ['from', 'to', 'states', 'to_states']:
            vals = np.array(df[col].values, dtype=object) if col in df.columns else np.full(len(df), None, dtype=object)
            blanks, codes = Mobility.by_value(vals, Mobility.blank, True)
            empty[col] = np.array(blanks)[codes]
            vals[empty[col]] = None
            out[col] = vals
        if (~empty.to_states & (empty['from'] | empty.to | empty.states)).any():
            errormsg = 'A row of a movement register with to_states moves people from its states to those states (in order, e.g. states "S1, I1" and to_states "S2, I2"), so it needs a from, a to (which can be the same place), and states'
            raise ValueError(errormsg)
        if (empty['from'] & empty.to).any():
            errormsg = 'Every row of a movement register needs a from, a to, or both'
            raise ValueError(errormsg)
        if (out['n'] < 0).any():
            errormsg = 'The numbers of people in a movement register cannot be negative'
            raise ValueError(errormsg)
        return out

    def events(self, strat, sim):
        """
        The register, resolved against the levels and the timeline of a sim

        Returns the step at whose start each event happens, the position of the place each leaves
        (-1 for none) and goes to (-1 for none), the number of people, and the states it names (None
        for the default), in the order of the table within each step. Events before the start or
        after the last step are dropped, with a note. Also the distinct pairs of `states` and
        `to_states` (`sets`, in the order they first appear), and which of them each event names (`set_of`).
        """
        df = self.data
        times = sc.dataframe(index=df['date'].values)
        sim_t, data_t = ss.calibration.data_times(sim, times)
        tol = 1e-9*max(np.abs(sim_t).max(initial=1), 1)
        ti = np.searchsorted(sim_t, data_t - tol, side='left') # The first step that starts on or after each date
        inside = (data_t >= sim_t[0] - tol) & (ti < len(sim_t))
        if not inside.all():
            ss.note(f'{(~inside).sum():n} of the {len(df):n} events in the movement register for "{strat.name}" fall outside the run, and are left out', sim=sim)
        labels = {str(label):i for i,label in enumerate(strat.labels)}
        def level(v):
            """ Places are given by label (a number read from a file with blanks in it is read as a float, so 5.0 is "5") """
            if Mobility.blank(v):
                return -1
            key = str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
            if key not in labels:
                errormsg = f'The movement register names a place "{v}", which is not a level of "{strat.name}"; places are given by label, e.g. {sc.strjoin(list(labels)[:3])}'
                raise KeyError(errormsg)
            return labels[key]
        order = np.argsort(ti[inside], kind='stable')
        rows = np.flatnonzero(inside)[order]
        col = lambda name: np.asarray(df[name].values, dtype=object)[rows] # As arrays, since reading a dataframe one cell at a time is slow
        def places(name):
            """ The position of the place in each entry of a column """
            vals, codes = Mobility.by_value(col(name), level, -1)
            return np.array(vals, dtype=np.int64)[codes]
        names = lambda v: None if Mobility.blank(v) else tuple(s.strip() for s in str(v).replace(';', ',').split(','))
        st_vals, st_codes = Mobility.by_value(col('states'), names, None)
        ts_vals, ts_codes = Mobility.by_value(col('to_states'), names, None)
        n_ts = len(ts_vals)
        set_of, pairs = pd.factorize((st_codes % len(st_vals))*n_ts + ts_codes % n_ts) # The distinct pairs of states and to_states, by their positions in st_vals and ts_vals (a code of -1 is the last, None)
        sets = [(st_vals[pair//n_ts], ts_vals[pair%n_ts]) for pair in pairs]
        for a,b in sets:
            if b is not None and len(a) != len(b):
                errormsg = f'A row of a movement register moves people from the states {a} to {b}, which need to be the same number'
                raise ValueError(errormsg)
        states, to_states = np.empty(len(sets), dtype=object), np.empty(len(sets), dtype=object) # Filled one at a time, since NumPy would read a list of tuples as rows
        for i,(a,b) in enumerate(sets):
            states[i], to_states[i] = a, b
        return sc.objdict(ti=ti[rows], src=places('from'), dst=places('to'), n=np.asarray(df['n'].values[rows], dtype=float),
                          states=states[set_of], to_states=to_states[set_of], sets=sets, set_of=set_of)

    def matrix(self, strat):
        """ The movement between each pair of levels, with nothing on the diagonal """
        if self.data is not None:
            errormsg = 'A movement register has no matrix of rates; its events are applied on their dates'
            raise ValueError(errormsg)
        out = self.model.matrix(strat) if self.model is not None else self.given.copy()
        if out.shape != (strat.n, strat.n):
            raise ss.ShapeError(f'the mobility matrix for "{strat.name}" has shape {out.shape}, but there are {strat.n} levels ({sc.strjoin(strat.labels)}).')
        if spsp.issparse(out):
            out = spsp.csr_array(out - spsp.diags_array(out.diagonal()))
            out.eliminate_zeros()
        else:
            np.fill_diagonal(out, 0)
        return out

    def time_share(self, strat):
        """ For commuting: the share of their time that people from each level (rows) spend in each level; rows sum to 1. A sparse array, since most pairs of places share no one """
        out = self.matrix(strat)
        total = np.asarray(out.sum(axis=1)).ravel()
        away = None if self.away is None else float(self.away) # A fitted ss.par, as its value
        if away is not None and self.scale == 'mean':
            scale = np.full(strat.n, away/total.mean() if total.mean() > 0 else 0.0)
        elif away is not None:
            scale = np.divide(away, total, out=np.zeros(strat.n), where=total>0)
        else: # Nobody can spend more than all their time away
            capped = (total > 1).sum()
            if capped:
                ss.warn(f'The movement for "{strat.name}" would have the people of {capped:n} of its {strat.n:n} places spend more than all their time away, so it was capped at all of it (nobody is ever at home there). Give the share of time away with away= instead.')
            scale = 1/np.maximum(total, 1)
        if self.cap is not None: # No place over the cap
            scale = np.minimum(scale, np.divide(float(self.cap), total, out=np.zeros(strat.n), where=total>0))
        if spsp.issparse(out):
            out = spsp.csr_array(out.multiply(scale[:, None]))
            out = spsp.csr_array(out + spsp.diags_array(np.maximum(1 - np.asarray(out.sum(axis=1)).ravel(), 0)))
        else:
            out = out*scale[:, None]
            np.fill_diagonal(out, np.maximum(1 - out.sum(axis=1), 0))
            out = spsp.csr_array(out)
        out.eliminate_zeros()
        out.sort_indices()
        return out

    def coupling(self, strat):
        """ For a moving force of infection: the share of the force of infection of each level (columns) that each level (rows) receives, as a sparse array """
        out = spsp.csr_array(self.time_share(strat).T) # Each place keeps its own share, and receives what each other place's force sends it
        out.sort_indices()
        return out
