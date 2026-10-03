"""
Networks that connect people within a population
"""
import inspect
import numpy as np
import numba as nb
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

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
    every swap, so the shuffle is *not* random-number safe -- matching `ss.RandomExactNet`'s
    intended behavior (see `ss.RandomSafeNet` for the CRN-safe version).
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
        contact_inds = ss.find_contacts(self.edges.p1, self.edges.p2, sc.toarray(inds), as_array=as_array)
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
        self.graph = graph
        self.define_pars(seed=True, p=None, n_contacts=10)
        self.graph_kwargs = {k:kwargs.pop(k) for k in list(kwargs) if k not in self.pars and k not in ss.modules.module_args} # Other arguments are passed to the graph generator
        self.update_pars(pars, **kwargs)
        self.dist = ss.Dist(name='StaticNet')
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.n_agents = sim.pars.n_agents
        if self.graph is None:
            import networkx as nx # Lazy import since slow
            self.graph = nx.fast_gnp_random_graph # Fast random (Erdos-Renyi) graph creator
        n_contacts = self.pars.pop('n_contacts') # Remove from pars dict, but use only if p is not supplied
        if self.pars.p is None: # Convert from n_contacts to probability
            self.pars.p = n_contacts/self.n_agents
        return

    def init_post(self):
        super().init_post()
        if 'seed' in self.pars and self.pars.seed is True:
            self.pars.seed = self.dist.rng
        if callable(self.graph):
            args = inspect.signature(self.graph).parameters
            kw = {k:v for k,v in self.pars.items() if k in args} # Only pass p and seed if the generator accepts them
            try:
                self.graph = self.graph(n=self.n_agents, **kw, **self.graph_kwargs)
            except TypeError as e:
                ss.warn(f"{str(e)}: networkx {self.graph.name} not supported. Try using ss.StaticNet() instead.")
                raise e
        self.validate_pop(self.n_agents)
        self.get_edges()
        return

    def validate_pop(self, n_agents):
        n_nodes = self.graph.number_of_nodes()
        if n_nodes > n_agents:
            errmsg = (f"Please ensure the number of nodes in graph ({n_nodes}) is less than "
                      f"or equal to (<=) the agent population size ({n_agents}).")
            raise ValueError(errmsg)

        if not self.graph.number_of_edges():
            errmsg = f"The nx generator {self.graph.name} produced a graph with no edges"
            raise ValueError(errmsg)

    def get_edges(self):
        p1s = []
        p2s = []
        for edge in self.graph.edges():
            p1, p2 = edge
            p1s.append(p1)
            p2s.append(p2)
        edges = dict(p1=p1s, p2=p2s, beta=np.ones_like(p1s))
        self.append(edges)
        return

    def step(self):
        pass


class RandomExactNet(DynamicNetwork):
    """
    Random connectivity between agents

    Args:
        n_contacts (int/`ss.Dist`): the average number of (bidirectional) contacts between agents
        dur (int/`ss.dur`): the duration of each contact
        beta (float): the default beta value for each edge
        age_range (list): if supplied, only agents with `age_range[0] <= age < age_range[1]` are included
        dynamic (bool): if True (default), edges are replaced once their duration ends; if False, edges are created once and never change
            (so agents who join later, e.g. newborns, are not added)

    Note: n_contacts = 10 will create *5* edges per agent. Since disease transmission
    usually occurs bidirectionally, this means that the effective number of contacts
    per agent is actually 10. Consider 3 agents with 3 edges between them (a triangle):
    each agent is connected to 2 other agents.

    Example:
        ```python
        # Static network with Poisson-distributed contacts among adults, as in Covasim v3
        net = ss.RandomNet(n_contacts=ss.poisson(20), age_range=[18, 65], dynamic=False)
        ```
    """
    def __init__(self, pars=None, n_contacts=_, dur=_, beta=_, age_range=_, dynamic=_, **kwargs):
        """ Initialize """
        super().__init__()
        self.define_pars(
            n_contacts = ss.constant(10),
            dur = ss.years(0), # Note; network edge durations are required to have the same unit as the network
            beta = 1.0,
            age_range = None,
            dynamic = True,
        )
        self.update_pars(pars, **kwargs)
        self.dist = ss.Dist(distname='RandomNet') # Default RNG; name kept as 'RandomNet' (not the class name) so results stay reproducible across the rename, and shared with the RandomNet subclass
        return

    def get_edges(self, inds, n_contacts):
        """
        Efficiently find edges

        Note that because of the shuffling operation, each person is assigned 2N contacts
        (i.e. if a person has 5 contacts, they appear 5 times in the 'source' array and 5
        times in the 'target' array). Therefore, the `n_contacts` argument to this
        function should be HALF of the total contacts a person is expected to have, if both
        the source and target array outputs are used (e.g. for social contacts)

        adjusted_number_of_contacts = np.round(n_contacts / 2).astype(ss.dtype.int)

        Whereas for asymmetric contacts (e.g. staff-public interactions) it might not be necessary

        Args:
            inds (list/array): person indices
            n_contacts (list/array): the same length as `inds` with the number of bidirectional contacts to assign to each person

        Returns:
            Two arrays, for source and target
        """
        source = np.repeat(inds, n_contacts)
        target = source.copy()
        randvals = self.dist.rng.random(len(target)) # Precompute the random values for the shuffle
        fisher_yates_shuffle(target, randvals) # Shuffle the target in place (faster equivalent of rng.permutation)
        self.dist.jump() # Reset the RNG manually; does not auto-jump since using rng directly above # TODO, think if there's a better way
        return source, target

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
            p1, p2 = self.get_edges(uids, n_conn)
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


class RandomNet(RandomExactNet):
    """
    A faster, approximate version of `ss.RandomExactNet`

    Identical to `ss.RandomExactNet` except that the target end of each edge is drawn
    by sampling the source stubs *with* replacement, rather than by shuffling them
    (i.e. sampling without replacement). This is roughly 1.8x faster at generating
    edges, but the per-agent degree is no longer fixed: the source side of each edge
    is still exactly `n_contacts`, while the target side is Poisson-distributed with
    mean `n_contacts`. Agents therefore have somewhat more variable numbers of
    contacts than with `ss.RandomExactNet`.

    Because targets are drawn from the stub list, an agent's chance of being picked
    as a target is proportional to its own `n_contacts`, exactly as in
    `ss.RandomExactNet`. This means the approximation preserves the mixing structure
    of risk-stratified networks (e.g. a small high-risk core with many contacts
    correctly forms most of its edges with itself); drawing targets uniformly over
    agents instead would destroy that clustering and bias the realized degrees.

    Alternatively, with `uniform_targets=True`, each target is chosen uniformly from all
    eligible agents, regardless of their own `n_contacts`, as in Covasim v3's random networks.
    This gives less variable numbers of contacts (e.g. with `n_contacts=ss.poisson(20)`, a
    variance of about 15, rather than 30 for the default or 20 for `ss.RandomExactNet`), but
    loses the mixing structure described above.

    Note: like `ss.RandomExactNet`, this is not random-number safe; see `ss.RandomSafeNet`
    for the CRN-safe (but slower) version.

    Args:
        n_contacts (int/`ss.Dist`): the average number of (bidirectional) contacts between agents
        dur (int/`ss.dur`): the duration of each contact
        beta (float): the default beta value for each edge
        age_range (list): if supplied, only agents with `age_range[0] <= age < age_range[1]` are included
        dynamic (bool): if True (default), edges are replaced once their duration ends; if False, edges are created once and never change
        uniform_targets (bool): if True, choose the target of each edge uniformly from the eligible agents, rather than in proportion to their number of contacts

    Example:
        ```python
        # Static network with Poisson-distributed contacts, as in Covasim v3
        net = ss.RandomNet(n_contacts=ss.poisson(20), dynamic=False, uniform_targets=True)
        ```
    """
    def __init__(self, pars=None, n_contacts=_, dur=_, beta=_, age_range=_, dynamic=_, uniform_targets=_, **kwargs):
        super().__init__()
        self.define_pars(
            uniform_targets = False,
        )
        self.update_pars(pars, **kwargs)
        return

    def get_edges(self, inds, n_contacts):
        """ Find edges by sampling the source stubs with replacement (see `ss.RandomExactNet.get_edges`) """
        source = np.repeat(inds, n_contacts)
        n = len(source)
        if n and self.pars.uniform_targets:
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
        self.dist.jump() # Reset the RNG manually, as in RandomExactNet.get_edges
        return source, target


class RandomSafeNet(DynamicNetwork):
    """
    Create a CRN-safe, O(N) random network

    This network is similar to `ss.RandomExactNet()`, but is random-number safe
    (i.e., the addition of a single new agent will not perturb the entire rest
    of the network). However, it is somewhat slower than `ss.RandomExactNet()`,
    so should be used where CRN safety is important (e.g., scenario analysis).

    Note: `ss.RandomExactNet` uses `n_contacts`, which is the total number of contacts
    per agent. `ss.RandomSafeNet` uses `n_edges`, which is the total number of
    *edges* per agent. Since contacts are usually bidirectional, n_contacts = 2*n_edges.
    For example, `ss.RandomExactNet(n_contacts=10)` will give (nearly) identical results
    to `ss.RandomSafeNet(n_edges=5)`. In addition, whereas `n_contacts` can be
    a distribution, `n_edges` can only be an integer.

    Args:
        n_edges (int): the average number of (bi-directional) edges between agents
        dur (int/`ss.dur`): the duration of each contact
        beta (float): the default beta value for each edge
    """
    def __init__(self, pars=None, n_edges=_, dur=_, beta=_, **kwargs):
        super().__init__()
        self.define_pars(
            n_edges = 5,
            dur = 0, # Note; network edge durations are required to have the same unit as the network
            beta = 1.0,
        )
        self.update_pars(pars, **kwargs)
        if sc.isnumber(self.pars.dur): self.pars.dur = ss.years(self.pars.dur) # Interpret numbers as years, as for ss.RandomExactNet (the default isn't ss.years(0) so a Dist is allowed)
        self.dist = ss.random(name='RandomSafeNet')
        self.dist.hash_dtype = np.float64 # Since float32 only gives 2^24 distinct values, which would mean many ties (e.g. 25% of edges with a million agents)
        return

    def rep_rand(self, uids, sort=True):
        """ Reproducible repeated random numbers """
        n_agents = len(uids)
        n_conn = self.pars.n_edges
        r_list = []
        for i in range(n_conn): # Necessary loop; not possible to vectorize given the behavior of the random number generator
            r = self.dist.rvs(uids)
            r_list.append(r)
        r_arr = np.array(r_list).flatten()
        inds = np.tile(np.arange(n_agents), n_conn)
        rr = np.array([inds, r_arr]).T
        if sort:
            order = np.argsort(r_arr)
            rr = rr[order,:]
        self.rr = rr
        self.rr_uids = ss.uids(uids) # Store the UIDs so form_pairs() can map stub positions back to agents
        return rr

    def form_pairs(self, debug=False):
        """ From a 2N input array, return 2N-2 nearest-neighbor pairs """
        rr = self.rr
        out = []
        agent = self.rr_uids[rr[:,0].astype(int)] # Map stub positions back to agent UIDs
        v     = rr[:,1]
        center = v[1:-1]
        p1_dist = abs(center - v[:-2])
        p2_dist = abs(center - v[2:])
        use_p1 = p1_dist <= p2_dist # Ties (which are common with many agents, since the random numbers are float32) go to the lower neighbor
        source = agent[1:-1]
        target = np.where(use_p1, agent[:-2], agent[2:]).view(ss.uids)

        # Store additional information for debugging
        if debug:
            dist = np.where(use_p1, p1_dist, p2_dist)
            out = sc.objdict()
            out.pairs = sorted(list(zip(source, target)))
            out.src = source
            out.trg = target
            out.dist = dist
            out.p1dist = p1_dist
            out.p2dist = p2_dist
            self.pairs = out
        return source,target

    def add_pairs(self):
        """ Generate edges """
        people = self.sim.people
        born = people.alive & (people.age > 0)

        # Get the random numbers
        self.rep_rand(born.uids)

        # Form the pairs
        p1, p2 = self.form_pairs()
        beta = np.full(len(p1), self.pars.beta, dtype=ss_float)

        if isinstance(self.pars.dur, ss.Dist):
            dur = self.pars.dur.rvs(p1)
        elif self.pars.dur == 0:
            dur = np.zeros(len(p1))
        else:
            dur = np.ones(len(p1))*self.pars.dur/self.t.dt # Other option would be np.full(len(p1), self.pars.dur.x), but this is harder to read

        self.append(p1=p1, p2=p2, beta=beta, dur=dur)
        return

    def plot_matrix(self, **kwargs):
        """ Plot the distance matrix used for forming pairs: only ±1 from the diagonal is used """
        kw = ss.plot_args(kwargs)
        v = self.rr[:,1]
        dm = abs(v[:, np.newaxis] - v)**2
        with ss.style(**kw.style):
            fig = plt.figure(**kw.fig)
            plt.imshow(dm)
        return ss.return_fig(fig)


class ClusterNet(Network):
    """
    Fully connected clusters of agents, e.g. households, schools, or workplaces

    Each agent (optionally, only those within an age range) is assigned to one
    cluster, with cluster sizes drawn from `cluster_size`, and every pair of agents
    within a cluster is connected. Agents are assigned to clusters in a random order
    (from a CRN-safe random number per agent), so clusters are not correlated with
    other networks. Based on `make_microstructured_contacts()` from Covasim v3: as
    there, clusters of size 0 are skipped, and the last cluster is truncated to fit.

    Args:
        cluster_size (int/`ss.Dist`): the size of each cluster; a number is the mean of a Poisson distribution
        age_range (list): if supplied, only agents with `age_range[0] <= age < age_range[1]` are included
        beta (float): the beta value for each edge
        dynamic (bool): if True, the clusters are recreated on each timestep; if False (default), they are created once and never change

    Example:
        ```python
        households = ss.ClusterNet(name='households', cluster_size=ss.poisson(3))
        sim = ss.Sim(diseases='sis', networks=households)
        sim.run()
        ```
    """
    def __init__(self, pars=None, cluster_size=_, age_range=_, beta=_, dynamic=_, **kwargs):
        super().__init__()
        self.define_pars(
            cluster_size = ss.poisson(lam=4),
            age_range = None,
            beta = 1.0,
            dynamic = False,
        )
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.FloatArr('cluster', label='Cluster ID'),
        )
        self.rng_order = ss.random() # Order in which agents are assigned to clusters
        self.n_clusters = 0
        return

    def step(self):
        """ If dynamic, recreate the clusters; otherwise, do nothing """
        if self.pars.dynamic:
            self.remove_edges(np.ones(len(self), dtype=bool))
            self.cluster.set_nan(self.cluster.auids)
            self.n_clusters = 0
            self.add_pairs()
        return

    def add_pairs(self):
        """ Assign agents to clusters, and connect every pair of agents within each cluster """
        uids = self.eligible(self.pars.age_range).uids
        uids = uids[np.argsort(self.rng_order.rvs(uids))] # Shuffle
        n = len(uids)
        if not n:
            return

        # Draw cluster sizes until there are enough to cover everyone
        sizes = np.empty(0, dtype=ss_int)
        while sizes.sum() < n:
            new = np.round(self.pars.cluster_size.rvs(n)).astype(ss_int) # n clusters is enough unless some have size 0
            if not np.any(new > 0): # Avoid an infinite loop
                errormsg = f'The cluster size distribution for {self.name} did not produce any clusters of size 1 or more: {self.pars.cluster_size}'
                raise ValueError(errormsg)
            sizes = np.concatenate([sizes, new[new > 0]])
        n_clusters = np.searchsorted(np.cumsum(sizes), n) + 1
        sizes = sizes[:n_clusters]
        sizes[-1] -= sizes.sum() - n # Truncate the last cluster
        self.cluster[uids] = self.n_clusters + np.repeat(np.arange(n_clusters), sizes)
        self.n_clusters += n_clusters

        # Connect every pair (i<j) within each cluster, looping over cluster sizes rather than clusters
        starts = np.cumsum(sizes) - sizes
        p1 = [np.empty(0, dtype=ss_int)]
        p2 = [np.empty(0, dtype=ss_int)]
        for size in np.unique(sizes[sizes > 1]):
            these = starts[sizes == size][:, None]
            i, j = np.triu_indices(size, k=1)
            p1.append((these + i).ravel())
            p2.append((these + j).ravel())
        p1 = uids[np.concatenate(p1)]
        p2 = uids[np.concatenate(p2)]
        beta = np.full(len(p1), self.pars.beta, dtype=ss_float)
        self.append(p1=p1, p2=p2, beta=beta)
        return


class HybridNet(Network):
    """
    Covasim v3's "hybrid" population: households, schools, workplaces, and community

    When added to a sim, this expands into four separate networks: households (`h`,
    an `ss.ClusterNet`), and schools (`s`), workplaces (`w`), and community (`c`), each
    an `ss.RandomNet` with Poisson-distributed contacts. As in Covasim v3, the networks
    are static, and schools and workplaces are random networks restricted by age.
    Each network can then be modified separately, e.g. `sim.networks.s`.

    The default `beta` values are Covasim v3's per-layer relative transmissibility,
    and are applied to the edges of each network; for per-network transmissibility
    on the disease instead, set them to 1 and use e.g. `ss.SIS(beta=dict(h=0.3, s=0.06, w=0.06, c=0.03))`.

    Args:
        household_size (int/`ss.Dist`): the size of each household; a number is the mean of a Poisson distribution
        contacts (dict): the mean number of contacts per agent in schools, workplaces, and community (numbers or `ss.Dist`s)
        beta (dict): the beta value for each edge in each network
        school_ages (list): the age range of agents in schools
        work_ages (list): the age range of agents in workplaces
        dynamic (bool): whether the networks are recreated on each timestep
        uniform_targets (bool): whether to choose the targets of edges in schools, workplaces, and community uniformly, as in Covasim v3 (see `ss.RandomNet`)

    Examples:
        ```python
        sim = ss.Sim(diseases='sis', networks=ss.HybridNet())
        sim.run()
        print(sim.networks.keys()) # ['h', 's', 'w', 'c']

        # Larger households and smaller schools; contacts and beta can be partial
        hybrid = ss.HybridNet(household_size=ss.poisson(3), contacts=dict(s=10))
        ```
    """
    def __init__(self, pars=None, household_size=_, contacts=_, beta=_, school_ages=_, work_ages=_, dynamic=_, uniform_targets=_, **kwargs):
        super().__init__()
        self.define_pars(
            household_size = ss.poisson(lam=2.0), # Covasim v3 default (contacts['h'])
            contacts = dict(s=20, w=16, c=20),
            beta = dict(h=3.0, s=0.6, w=0.6, c=0.3),
            school_ages = [6, 22],
            work_ages = [22, 65],
            dynamic = False,
            uniform_targets = False,
        )
        defaults = sc.dcp(sc.objdict(contacts=self.pars.contacts, beta=self.pars.beta))
        self.update_pars(pars, **kwargs)
        for key,default in defaults.items(): # Allow partial dicts, e.g. contacts=dict(s=10)
            self.pars[key] = sc.mergedicts(default, self.pars[key])
        return

    def expand(self):
        """ Create the four networks; called automatically when added to a sim """
        p = self.pars
        n = {k: ss.poisson(lam=v) if sc.isnumber(v) else v for k,v in p.contacts.items()}
        kw = dict(dynamic=p.dynamic, dt=self.t.dt, start=self.t.start, stop=self.t.stop) # Pass the timeline to each network # TODO: make this more general for one module creating another
        rkw = dict(uniform_targets=p.uniform_targets, **kw)
        networks = [
            ClusterNet(name='h', label='Households', cluster_size=p.household_size, beta=p.beta['h'], **kw),
            RandomNet(name='s', label='Schools', n_contacts=n['s'], age_range=p.school_ages, beta=p.beta['s'], **rkw),
            RandomNet(name='w', label='Workplaces', n_contacts=n['w'], age_range=p.work_ages, beta=p.beta['w'], **rkw),
            RandomNet(name='c', label='Community', n_contacts=n['c'], beta=p.beta['c'], **rkw),
        ]
        return networks

    def step(self):
        """ Not used, since the networks returned by expand() are used instead """
        errormsg = 'ss.HybridNet() should be added to a sim, which replaces it with the networks from expand()'
        raise RuntimeError(errormsg)


class MFNet(SexualNetwork):
    """
    This network is built by **randomly pairing** males and female with variable
    relationship durations.

    Args:
        duration (`ss.Dist`): Can vary by age, year, and individual pair. Set scale=exp(mu) and s=sigma where mu,sigma are of the underlying normal distribution.
        debut (`ss.Dist`): Age of debut can vary by using callable parameter values
        acts (`ss.Dist`): Number of acts per year
        participation (`ss.Dist`): Probability of participating in this network - can vary by individual properties (age, sex, ...) using callable parameter values
        rel_part_rates (float): Relative participation in the network, i.e. a multiplier (from 0 to 1) on the probability of participating
    """
    def __init__(self, pars=None, duration=_, debut=_, acts=_, participation=_, rel_part_rates=_, **kwargs):
        super().__init__()
        self.define_pars(
            duration = ss.lognorm_ex(mean=ss.years(15), std=ss.years(1)),  # Can vary by age, year, and individual pair. Set scale=exp(mu) and s=sigma where mu,sigma are of the underlying normal distribution.
            debut = ss.normal(loc=16),  # Age of debut can vary by using callable parameter values
            acts = ss.poisson(lam=ss.freqperyear(80)),
            participation = ss.bernoulli(p=0.9),  # Probability of participating in this network - can vary by individual properties (age, sex, ...) using callable parameter values
            rel_part_rates = 1.0,
        )
        self.update_pars(pars, **kwargs)

        # Finish initialization
        self.dist = ss.choice(name='MFNet', replace=False) # Set the array later
        self.rel_part = ss.bernoulli(p=1.0) # Set p later, in case rel_part_rates is modified
        return

    def init_post(self):
        self.set_network_states()
        super().init_post()
        return

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
            dur_vals = self.pars.duration.rvs(p1)
            act_vals = self.pars.acts.rvs(p1)
        else: # TODO: rethink explanation without multirng
            # If multirng is enabled, we're here because some individuals in p1
            # are starting multiple relationships on this timestep. If using
            # slotted draws, as above, repeated relationships will get the same
            # duration and act rates, which is scientifically undesirable.
            # Instead, we fall back to a not-CRN safe approach:
            dur_vals = self.pars.duration.rvs(len(p1))  # Just use len(p1) to say how many draws are needed
            act_vals = self.pars.acts.rvs(len(p1))

        self.append(p1=p1, p2=p2, beta=beta, dur=dur_vals, acts=act_vals)

        return len(p1)

    def step(self):
        self.end_pairs()
        self.set_network_states(upper_age=self.t.dt.years) # Assign debut age and participation to agents born since the last step, i.e. those younger than one timestep; ages are in years, so the timestep must be too
        self.add_pairs()
        return


class MSMNet(SexualNetwork):
    """
    A network that randomly pairs males

    Args:
        duration (`ss.Dist`): Can vary by age, year, and individual pair. Set scale=exp(mu) and s=sigma where mu,sigma are of the underlying normal distribution.
        debut (`ss.Dist`): Age of debut can vary by using callable parameter values
        acts (`ss.Dist`): Number of acts per year
        participation (`ss.Dist`): Probability of participating in this network - can vary by individual properties (age, sex, ...) using callable parameter values
    """
    def __init__(self, pars=None, duration=_, debut=_, acts=_, participation=_, **kwargs):
        super().__init__()
        self.define_pars(
            duration = ss.lognorm_ex(mean=ss.years(2), std=ss.years(1)),
            debut = ss.normal(loc=16, scale=2),
            acts = ss.lognorm_ex(mean=ss.freqperyear(80), std=ss.freqperyear(20)),
            participation = ss.bernoulli(p=0.1),
        )
        self.update_pars(pars, **kwargs)
        self.dist = ss.random(name='MSMNet') # For shuffling available males before pairing
        return

    def init_post(self):
        self.set_network_states()
        super().init_post()
        return

    def set_network_states(self, upper_age=None):
        """ Set network states including age of entry into network and participation rates """
        people = self.sim.people
        if upper_age is None: uids = people.uid[people.male]
        else: uids = people.uid[people.male & (people.age < upper_age)]

        # Participation
        self.participant[people.female] = False
        self.participant[uids] = self.pars.participation.rvs(uids) # Should be CRN safe?

        # Debut
        self.debut[uids] = self.pars.debut.rvs(len(uids)) # Just pass len(uids) as this network is not crn safe anyway
        return

    def add_pairs(self):
        """ Pair all unpartnered MSM """
        available_m = self.available(self.sim.people, 'male')
        available_m = available_m[np.argsort(self.dist.rvs(available_m))] # Shuffle so pairing is random
        n_pairs = int(len(available_m)/2)
        p1 = available_m[:n_pairs]
        p2 = available_m[n_pairs:n_pairs*2]

        # Figure out durations
        if (len(p1) == len(np.unique(p1))):
            # No duplicates, so use slotting based on p1
            dur = self.pars.duration.rvs(p1)
            act_vals = self.pars.acts.rvs(p1)
        else:
            dur = self.pars.duration.rvs(len(p1)) # Just use len(p1) to say how many draws are needed
            act_vals = self.pars.acts.rvs(len(p1))

        self.append(p1=p1, p2=p2, beta=np.ones_like(p1), dur=dur, acts=act_vals)

        return len(p1)

    def step(self):
        self.end_pairs()
        self.set_network_states(upper_age=self.t.dt.years)
        self.add_pairs()
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
    """ Alias for `PrenatalNet`, kept for backwards compatibility """
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
                    n_contacts = ss.poisson(lam=n_contacts)
                name = f'pool:{sk}->{dk}'
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
        return

    def __len__(self):
        try:
            return len(self.pars.dst)
        except:
            return 0

    def init_post(self):
        super().init_post()

        if len(self.pars.diseases) == 0:
            self.diseases = [d for d in self.sim.diseases.values() if isinstance(d, ss.Infection)] # Assume the user wants all communicable diseases
        else:
            self.diseases = []
            for d in self.pars.diseases:
                if not isinstance(d, str):
                    raise TypeError(f'Diseases can be specified as ss.Disease objects or strings, not {type(d)}')
                if d not in self.sim.diseases:
                    raise KeyError(f'Could not find disease with name {d} in the list of diseases.')

                dis = self.sim.diseases[d]
                if not isinstance(dis, ss.Infection):
                    raise TypeError(f'Cannot create a mixing pool for disease {d}. Mixing pools only work for communicable diseases.')
                self.diseases.append(dis)

            if len(self.diseases) == 0:
                raise ValueError('You must specify at least one transmissible disease to use mixing pools')
        return

    def get_uids(self, func_or_array):
        if func_or_array is None:
            return self.sim.people.auids
        elif callable(func_or_array):
            return func_or_array(self.sim)
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

