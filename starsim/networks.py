"""
Networks that connect people within a population
"""
import networkx as nx
import numpy as np
import numba as nb
import sciris as sc
import starsim as ss
import scipy.optimize as spo
import scipy.spatial as spsp

# This has a significant impact on runtime, surprisingly
ss_float_ = ss.dtypes.float
ss_int_ = ss.dtypes.int


# Specify all externally visible functions this file defines; see also more definitions below
__all__ = ['Route', 'Network', 'DynamicNetwork', 'SexualNetwork']

# %% General network classes

class Route(ss.Module):
    """
    A transmission route -- e.g., a network, mixing pool, environmental transmission, etc.
    """
    pass


class Network(Route):
    """
    A class holding a single network of contact edges (connections) between people
    as well as methods for updating these.

    The input is typically arrays including: person 1 of the connection, person 2 of
    the connection, the weight of the connection, the duration and start/end times of
    the connection.

    Args:
        p1 (array): an array of length N, the number of connections in the network, with the indices of people on one side of the connection.
        p2 (array): an array of length N, the number of connections in the network, with the indices of people on the other side of the connection.
        beta (array): an array representing relative transmissibility of each connection for this network - TODO, do we need this?
        label (str): the name of the network (optional)
        kwargs (dict): other keys copied directly into the network

    Note that all arguments (except for label) must be arrays of the same length,
    although not all have to be supplied at the time of creation (they must all
    be the same at the time of initialization, though, or else validation will fail).

    **Examples**::

        # Generate an average of 10 contacts for 1000 people
        n_contacts_pp = 10
        n_people = 1000
        n = n_contacts_pp * n_people
        p1 = np.random.randint(n_people, size=n)
        p2 = np.random.randint(n_people, size=n)
        beta = np.ones(n)
        network = ss.Network(p1=p1, p2=p2, beta=beta, label='rand')
        network = ss.Network(dict(p1=p1, p2=p2, beta=beta), label='rand') # Alternate method

        # Convert one network to another with extra columns
        index = np.arange(n)
        self_conn = p1 == p2
        network2 = ss.Network(**network, index=index, self_conn=self_conn, label=network.label)
    """

    def __init__(self, key_dict=None, prenatal=False, postnatal=False, name=None, label=None, **kwargs):
        # Initialize as a module
        super().__init__(name=name, label=label)

        # Each relationship is characterized by these default set of keys, plus any user- or network-supplied ones
        default_keys = sc.objdict(
            p1 = ss_int_,
            p2 = ss_int_,
            beta = ss_float_,
        )
        self.meta = sc.mergedicts(default_keys, key_dict)
        self.prenatal = prenatal  # Prenatal connections are added at the time of conception. Requires ss.Pregnancy()
        self.postnatal = postnatal  # Postnatal connections are added at the time of delivery. Requires ss.Pregnancy()

        # Initialize the keys of the network
        self.edges = sc.objdict()
        for key, dtype in self.meta.items():
            self.edges[key] = np.empty((0,), dtype=dtype)

        # Set data, if provided
        for key, value in kwargs.items():
            self.edges[key] = np.array(value, dtype=self.meta.get(key)) # Overwrite dtype if supplied, else keep original
            self.initialized = True

        # Define states using placeholder values
        self.participant = ss.BoolArr('participant')
        self.validate_uids()
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

    def init_post(self, add_pairs=True):
        super().init_post()
        if add_pairs: self.add_pairs()
        return

    def __len__(self):
        """ The length is the number of edges """
        try:    return len(self.edges.p1)
        except: return 0

    def __repr__(self, **kwargs):
        """ Convert to a dataframe for printing """
        try:
            namestr = self.name
            labelstr = f'"{self.label}"' if self.label else '<no label>'
            keys_str = ', '.join(self.edges.keys())
            output = f'{namestr}({labelstr}, {keys_str})\n'  # e.g. Network("r", p1, p2, beta)
            output += self.to_df().__repr__()
        except:
            output = sc.prepr(self, vals=False)
        return output

    def __contains__(self, item):
        """
        Check if a person is present in a network

        Args:
            item: Person index

        Returns: True if person index appears in any interactions
        """
        return (item in self.edges.p1) or (item in self.edges.p2) # TODO: chek if (item in self.members) is faster

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
        self.results['n_edges'][self.ti] = len(self)
        return

    def to_graph(self): # pragma: no cover
        """
        Convert to a networkx DiGraph

        **Example**::

            import networkx as nx
            sim = ss.Sim(n_agents=100, networks='mf').init()
            G = sim.networks.randomnet.to_graph()
            nx.draw(G)
        """
        keys = [('p1', int), ('p2', int), ('beta', float)]
        data = [np.array(self.edges[k], dtype=dtype).tolist() for k,dtype in keys]
        G = nx.DiGraph()
        G.add_weighted_edges_from(zip(*data), weight='beta')
        nx.set_edge_attributes(G, self.label, name='layer')
        return G

    def to_dict(self):
        """ Convert to dictionary """
        d = {k: self.edges[k] for k in self.meta_keys()}
        return d

    def to_df(self):
        """ Convert to dataframe """
        df = sc.dataframe.from_dict(self.to_dict())
        return df

    def from_df(self, df, keys=None):
        """ Convert from a dataframe """
        if keys is None:
            keys = self.meta_keys()
        for key in keys:
            self.edges[key] = df[key].to_numpy()
        return self

    def find_contacts(self, inds, as_array=True):
        """
        Find all contacts of the specified people

        For some purposes (e.g. contact tracing) it's necessary to find all the edges
        associated with a subset of the people in this network. Since edges are bidirectional
        it's necessary to check both p1 and p2 for the target indices. The return type is a Set
        so that there is no duplication of indices (otherwise if the Network has explicit
        symmetric interactions, they could appear multiple times). This is also for performance so
        that the calling code doesn't need to perform its own unique() operation. Note that
        this cannot be used for cases where multiple connections count differently than a single
        infection, e.g. exposure risk.

        Args:
            inds (array): indices of people whose edges to return
            as_array (bool): if true, return as sorted array (otherwise, return as unsorted set)

        Returns:
            contact_inds (array): a set of indices for pairing partners

        Example: If there were a network with
        - p1 = [1,2,3,4]
        - p2 = [2,3,1,4]
        Then find_edges([1,3]) would return {1,2,3}
        """

        # Check types
        if not isinstance(inds, np.ndarray):
            inds = sc.promotetoarray(inds)
        if inds.dtype != np.int64:  # pragma: no cover # This is int64 since indices often come from utils.true(), which returns int64
            inds = np.array(inds, dtype=np.int64)

        # Find the edges
        contact_inds = ss.find_contacts(self.edges.p1, self.edges.p2, inds)
        if as_array:
            contact_inds = np.fromiter(contact_inds, dtype=ss_int_)
            contact_inds.sort()

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
        keep = ~(np.isin(self.edges.p1, uids) | np.isin(self.edges.p2, uids))
        for k in self.meta_keys():
            self.edges[k] = self.edges[k][keep]

        return

    def net_beta(self, disease_beta=None, inds=None, disease=None):
        """ Calculate the beta for the given disease and network """
        if inds is None: inds = Ellipsis
        return self.edges.beta[inds] * disease_beta # Beta should already include dt if desired


class DynamicNetwork(Network):
    """ A network where partnerships update dynamically """
    def __init__(self, key_dict=None, **kwargs):
        key_dict = sc.mergedicts({'dur': ss_float_}, key_dict)
        super().__init__(key_dict=key_dict, **kwargs)
        return

    def step(self):
        self.end_pairs()
        self.add_pairs()
        return

    def end_pairs(self):
        people = self.sim.people
        self.edges.dur = self.edges.dur - self.t.dt # TODO: think about whether this is right # Update: it is, if duration is *NOT* a ss.dur! Otherwise it should be -1, in timestep units

        # Non-alive agents are removed
        active = (self.edges.dur > 0) & people.alive[self.edges.p1] & people.alive[self.edges.p2]
        for k in self.meta_keys():
            self.edges[k] = self.edges[k][active]
        return len(active)


class SexualNetwork(DynamicNetwork):
    """ Base class for all sexual networks """
    def __init__(self, key_dict=None, **kwargs):
        key_dict = sc.mergedicts({'acts': ss_int_}, key_dict)
        super().__init__(key_dict=key_dict, **kwargs)
        self.debut = ss.FloatArr('debut', default=0)
        return

    def active(self, people):
        # Exclude people who are not alive
        valid_age = people.age > self.debut
        active = self.participant & valid_age & people.alive
        return active

    def available(self, people, sex):
        # Currently assumes unpartnered people are available
        # Could modify this to account for concurrency
        # This property could also be overwritten by a NetworkConnector
        # which could incorporate information about membership in other
        # contact networks
        available = people[sex] & self.active(people)
        available[self.edges.p1] = False
        available[self.edges.p2] = False
        return available.uids

    def net_beta(self, disease_beta=None, inds=None, disease=None):
        if inds is None: inds = Ellipsis
        return self.edges.beta[inds] * (1 - (1 - disease_beta) ** (self.edges.acts[inds] * self.t.dt))


# %% Specific instances of networks
__all__ += ['StaticNet', 'RandomNet', 'ErdosRenyiNet', 'DiskNet', 'NullNet', 'MFNet', 'MSMNet', 'EmbeddingNet', 'MaternalNet', 'PrenatalNet', 'PostnatalNet']


class StaticNet(Network):
    """
    A network class of static partnerships converted from a networkx graph. There's no formation of new partnerships
    and initialized partnerships only end when one of the partners dies. The networkx graph can be created outside Starsim
    if population size is known. Or the graph can be created by passing a networkx generator function to Starsim.

    If "seed=True" is passed as a keyword argument or a parameter in pars, it is replaced with the built-in RNG.
    The parameter "n" is supplied automatically to be equal to n_agents.

    **Examples**::
        # Generate a networkx graph and pass to Starsim
        import networkx as nx
        import starsim as ss
        g = nx.scale_free_graph(n=10000)
        ss.StaticNet(graph=g)

        # Pass a networkx graph generator to Starsim
        ss.StaticNet(graph=nx.erdos_renyi_graph, p=0.0001, seed=True)

        # Just create a default graph
    """

    def __init__(self, graph=None, pars=None, **kwargs):
        super().__init__()
        self.graph = graph
        self.define_pars(seed=True, p=None, n_contacts=10)
        self.update_pars(pars, **kwargs)
        self.dist = ss.Dist(name='StaticNet')
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.n_agents = sim.pars.n_agents
        if self.graph is None:
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
            try:
                self.graph = self.graph(n=self.n_agents, **self.pars)
            except TypeError as e:
                print(f"{str(e)}: networkx {self.graph.name} not supported. Try using ss.NullNet().")
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


class RandomNet(DynamicNetwork):
    """ Random connectivity between agents """

    def __init__(self, pars=None, key_dict=None, **kwargs):
        """ Initialize """
        super().__init__(key_dict=key_dict)
        self.define_pars(
            n_contacts = ss.constant(10),
            dur = 0, # Note; network edge durations are required to have the same unit as the network
        )
        self.update_pars(pars, **kwargs)
        self.dist = ss.Dist(distname='RandomNet') # Default RNG
        return

    def init_post(self):
        self.add_pairs()
        return

    @staticmethod
    @nb.njit(fastmath=True, parallel=False, cache=True)
    def get_source(inds, n_contacts):
        """ Optimized helper function for getting contacts """
        total_number_of_half_edges = np.sum(n_contacts)
        count = 0
        source = np.zeros((total_number_of_half_edges,), dtype=ss_int_)
        for i, person_id in enumerate(inds):
            n = n_contacts[i]
            source[count: count + n] = person_id
            count += n
        return source

    def get_edges(self, inds, n_contacts):
        """
        Efficiently find edges

        Note that because of the shuffling operation, each person is assigned 2N contacts
        (i.e. if a person has 5 contacts, they appear 5 times in the 'source' array and 5
        times in the 'target' array). Therefore, the `number_of_contacts` argument to this
        function should be HALF of the total contacts a person is expected to have, if both
        the source and target array outputs are used (e.g. for social contacts)

        adjusted_number_of_contacts = np.round(number_of_contacts / 2).astype(ss.dtype.int)

        Whereas for asymmetric contacts (e.g. staff-public interactions) it might not be necessary

        Args:
            inds: List/array of person indices
            number_of_contacts: List/array the same length as `inds` with the number of unidirectional
            contacts to assign to each person. Therefore, a person will have on average TWICE this number
            of random contacts.

        Returns: Two arrays, for source and target
        """
        source = self.get_source(inds, n_contacts)
        target = self.dist.rng.permutation(source)
        self.dist.jump() # Reset the RNG manually; does not auto-jump since using rng directly above # TODO, think if there's a better way
        return source, target

    def add_pairs(self):
        """ Generate edges """
        people = self.sim.people
        born = people.alive & (people.age > 0)
        if isinstance(self.pars.n_contacts, ss.Dist):
            number_of_contacts = self.pars.n_contacts.rvs(born.uids)  # or people.uid?
        else:
            number_of_contacts = np.ones(len(people))*self.pars.n_contacts

        number_of_contacts = sc.randround(number_of_contacts / 2).astype(ss_int_)  # One-way contacts

        p1, p2 = self.get_edges(born.uids, number_of_contacts)
        beta = np.ones(len(p1), dtype=ss_float_)

        if isinstance(self.pars.dur, ss.Dist):
            dur = self.pars.dur.rvs(p1)
        else:
            dur = np.ones(len(p1))*self.pars.dur # Other option would be np.full(len(p1), self.pars.dur.x), but this is harder to read

        self.append(p1=p1, p2=p2, beta=beta, dur=dur)
        return


class ErdosRenyiNet(DynamicNetwork):
    """
    In the Erdos-Renyi network, every possible edge has a probability, p, of
    being created on each time step.

    The degree of each node will have a binomial distribution, considering each
    of the N-1 possible edges connection this node to the others will be created
    with probability p.

    Please be careful with the `dur` parameter. When set to 0, new edges will be
    created on each time step. If positive, edges will persist for `dur` years.
    Note that the existence of edges from previous time steps will not prevent
    or otherwise alter the creation of new edges on each time step, edges will
    accumulate over time.
    """

    def __init__(self, pars=None, key_dict=None, **kwargs):
        """ Initialize """
        super().__init__(key_dict=key_dict)
        self.define_pars(
            p = 0.1, # Probability of each edge
            dur = 0, # Duration of zero ensures that new random edges are formed on each time step
        )
        self.update_pars(pars, **kwargs)
        self.randint = ss.randint(low=np.iinfo('int64').min, high=np.iinfo('int64').max, dtype=np.int64) # Used to draw a random number for each agent as part of creating edges
        return

    def init_post(self):
        self.add_pairs()
        return

    def add_pairs(self):
        """ Generate contacts """
        people = self.sim.people
        born_uids = (people.age > 0).uids

        # Sample integers
        ints = self.randint.rvs(born_uids)

        # All possible edges are upper triangle of complete matrix
        idx1, idx2 = np.triu_indices(n=len(born_uids), k=1)

        # Use integers to create random numbers per edge
        i1 = ints[idx1]
        i2 = ints[idx2]
        r = ss.utils.combine_rands(i1, i2) # TODO: use ss.multi_rand()
        edge = r <= self.pars.p

        p1 = idx1[edge]
        p2 = idx2[edge]
        beta = np.ones(len(p1), dtype=ss_float_)

        if isinstance(self.pars.dur, ss.Dist):
            dur = self.pars.dur.rvs(p1)
        else:
            dur = np.ones(len(p1))*self.pars.dur

        self.append(p1=p1, p2=p2, beta=beta, dur=dur)
        return


class DiskNet(Network):
    """
    Disk graph in which edges are made between agents located within a user-defined radius.

    Interactions take place within a square with edge length of 1. Agents are
    initialized to have a random position and orientation within this square. On
    each time step, agents advance v*dt in the direction they are pointed. When
    encountering a wall, agents are reflected.

    Edges are formed between two agents if they are within r distance of each other.
    """

    def __init__(self, pars=None, key_dict=None, **kwargs):
        """ Initialize """
        super().__init__(key_dict=key_dict)
        self.define_pars(
            r = 0.1, # Radius
            v = 0.05, # Velocity
        )
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.FloatArr('x', default=ss.random(), label='X position'),
            ss.FloatArr('y', default=ss.random(), label='Y position'),
            ss.FloatArr('theta', default=ss.uniform(high=2*np.pi), label='Heading'),
        )
        return

    def init_post(self):
        self.add_pairs()
        return

    def step(self):
        # Motion step
        vdt = self.pars.v * self.t.dt
        self.x[:] = self.x + vdt * np.cos(self.theta)
        self.y[:] = self.y + vdt * np.sin(self.theta)

        # Wall bounce

        ## Right edge
        inds = (self.x > 1).uids
        self.x[inds] = 2 - self.x[inds]
        self.theta[inds] = np.pi - self.theta[inds]

        ## Left edge
        inds = (self.x < 0).uids
        self.x[inds] =  -self.x[inds]
        self.theta[inds] = np.pi - self.theta[inds]

        ## Top edge
        inds = (self.y > 1).uids
        self.y[inds] = 2 - self.y[inds]
        self.theta[inds] = - self.theta[inds]

        ## Bottom edge
        inds = (self.y < 0).uids
        self.y[inds] = -self.y[inds]
        self.theta[inds] = - self.theta[inds]

        self.add_pairs()
        return

    def add_pairs(self):
        """ Generate contacts """
        p1, p2 = np.triu_indices(n=len(self.x), k=1)
        d12_sq = (self.x.raw[p2]-self.x.raw[p1])**2 + (self.y.raw[p2]-self.y.raw[p1])**2
        edge = d12_sq < self.pars.r**2

        self.edges['p1'] = ss.uids(p1[edge])
        self.edges['p2'] = ss.uids(p2[edge])
        self.edges['beta'] = np.ones(len(self.p1), dtype=ss_float_)

        return


class NullNet(Network):
    """
    A convenience class for a network of size n that only has self-connections with a weight of 0.
    This network can be useful for debugging purposes or as a placeholder network during development
    for conditions that require more complex network mechanisms.

    Guarantees there's one (1) contact per agent (themselves), and that their connection weight is zero.

    For an empty network (ie, no edges) use
    >> import starsim as ss
    >> import networkx as nx
    >> empty_net_static = ss.StaticNet(nx.empty_graph)
    >> empty_net_rand = ss.RandomNet(n_contacts=0)

    """

    def __init__(self, n_people=None, **kwargs):
        self.n = n_people
        super().__init__(**kwargs)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        popsize = sim.pars['n_agents']
        if self.n is None:
            self.n = popsize
        else:
            if self.n > popsize:
                errormsg = f'Please ensure the size of the network ({self.n} is less than or equal to the population size ({popsize}).'
                raise ValueError(errormsg)
        self.get_edges()
        return

    def get_edges(self):
        indices = np.arange(self.n)
        self.append(dict(p1=indices, p2=indices, beta=np.zeros_like(indices)))
        return

    def step(self):
        """ Not used for NullNet """
        pass


class MFNet(SexualNetwork):
    """
    This network is built by **randomly pairing** males and female with variable
    relationship durations.
    """
    def __init__(self, pars=None, key_dict=None, **kwargs):
        super().__init__(key_dict=key_dict)
        self.define_pars(
            duration = ss.lognorm_ex(mean=15),  # Can vary by age, year, and individual pair. Set scale=exp(mu) and s=sigma where mu,sigma are of the underlying normal distribution.
            participation = ss.bernoulli(p=0.9),  # Probability of participating in this network - can vary by individual properties (age, sex, ...) using callable parameter values
            debut = ss.normal(loc=16),  # Age of debut can vary by using callable parameter values
            acts = ss.poisson(lam=80), # TODO: make this work with ss.rate, which it currently does not due to network initialization limitations
            rel_part_rates = 1.0,
        )
        self.update_pars(pars=pars, **kwargs)

        # Finish initialization
        self.dist = ss.choice(name='MFNet', replace=False) # Set the array later
        return

    def init_post(self):
        self.set_network_states()
        self.add_pairs()
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
        self.participant[uids] = self.pars.participation.rvs(uids)
        return

    def set_debut(self, upper_age=None):
        """ Set debut age """
        people = self.sim.people
        if upper_age is None: uids = people.auids
        else: uids = (people.age < upper_age).uids
        self.debut[uids] = self.pars.debut.rvs(uids)
        return

    def add_pairs(self):
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
        self.set_network_states(upper_age=self.t.dt) # TODO: check
        self.add_pairs()
        return


class MSMNet(SexualNetwork):
    """
    A network that randomly pairs males
    """

    def __init__(self, pars=None, key_dict=None, **kwargs):
        super().__init__(key_dict=key_dict)
        self.define_pars(
            duration = ss.lognorm_ex(mean=2, std=1),
            debut = ss.normal(loc=16, scale=2),
            acts = ss.lognorm_ex(mean=80, std=20),
            participation = ss.bernoulli(p=0.1),
        )
        self.update_pars(pars, **kwargs)
        return

    def init_post(self):
        self.set_network_states()
        self.add_pairs()
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
        self.set_network_states()
        self.add_pairs()
        return


class EmbeddingNet(MFNet):
    """
    Heterosexual age-assortative network based on a one-dimensional embedding. Could be made more generic.
    """

    def __init__(self, pars=None, **kwargs):
        """
        Create a sexual network from a 1D embedding based on age

        Args:
            male_shift is the average age that males are older than females in partnerships
        """
        super().__init__()
        self.define_pars(
            inherit = True, # The MFNet already comes with pars, we want to keep those
            embedding_func = ss.normal(name='EmbeddingNet', loc=self.embedding_loc, scale=2),
            male_shift = 5,
        )
        self.update_pars(pars, **kwargs)
        return

    @staticmethod
    def embedding_loc(module, sim, uids):
        loc = sim.people.age[uids]
        loc[sim.people.female[uids]] += module.pars.male_shift  # Shift females so they will be paired with older men
        return loc

    def add_pairs(self):
        people = self.sim.people
        available_m = self.available(people, 'male')
        available_f = self.available(people, 'female')

        if not len(available_m) or not len(available_f):
            if ss.options.verbose > 1:
                print('No pairs to add')
            return 0

        available = ss.uids.cat(available_m, available_f)
        loc = self.pars.embedding_func.rvs(available)
        loc_f = loc[people.female[available]]
        loc_m = loc[~people.female[available]]

        dist_mat = spsp.distance_matrix(loc_m[:, np.newaxis], loc_f[:, np.newaxis])
        ind_m, ind_f = spo.linear_sum_assignment(dist_mat)
        n_pairs = len(ind_f)

        # Finalize pairs
        p1 = available_m[ind_m]
        p2 = available_f[ind_f]
        beta = np.ones(n_pairs) # TODO: Allow custom beta
        dur_vals = self.pars.duration.rvs(p1)
        act_vals = self.pars.acts.rvs(p1)

        self.append(p1=p1, p2=p2, beta=beta, dur=dur_vals, acts=act_vals)
        return len(beta)


class MaternalNet(DynamicNetwork):
    """
    Base class for maternal transmission
    Use PrenatalNet and PostnatalNet to capture transmission in different phases
    """
    def __init__(self, key_dict=None, prenatal=True, postnatal=False, **kwargs):
        """
        Initialized empty and filled with pregnancies throughout the simulation
        """
        key_dict = sc.mergedicts(dict(dur=ss_float_, start=ss_int_, end=ss_int_), key_dict)
        super().__init__(key_dict=key_dict, prenatal=prenatal, postnatal=postnatal, **kwargs)
        return

    def step(self):
        """
        Set beta to 0 for women who complete duration of transmission
        Keep connections for now, might want to consider removing

        NB: add_pairs() and end_pairs() are NOT called here; this is done separately
        in ss.Pregnancy.update_states().
        """
        inactive = self.edges.end <= self.ti
        self.edges.beta[inactive] = 0
        return

    def end_pairs(self):
        people = self.sim.people
        edges = self.edges
        active = (edges.end > self.ti) & people.alive[edges.p1] & people.alive[edges.p2]
        for k in self.meta_keys():
            edges[k] = edges[k][active]
        return len(active)

    def add_pairs(self, mother_inds=None, unborn_inds=None, dur=None, start=None):
        """ Add connections between pregnant women and their as-yet-unborn babies """
        if mother_inds is None:
            return 0
        else:
            if start is None:
                start = np.ones_like(dur)*self.ti
            n = len(mother_inds)
            beta = np.ones(n)
            end = start + sc.promotetoarray(dur)
            self.append(p1=mother_inds, p2=unborn_inds, beta=beta, dur=dur, start=start, end=end)
            return n


class PrenatalNet(MaternalNet):
    """ Prenatal transmission network """
    def __init__(self, key_dict=None, prenatal=True, postnatal=False, **kwargs):
        super().__init__(key_dict=key_dict, prenatal=prenatal, postnatal=postnatal, **kwargs)
        return

class PostnatalNet(MaternalNet):
    """ Postnatal transmission network """
    def __init__(self, key_dict=None, prenatal=False, postnatal=True, **kwargs):
        super().__init__(key_dict=key_dict, prenatal=prenatal, postnatal=postnatal, **kwargs)
        return


__all__ += ['AgeGroup', 'MixingPools', 'MixingPool']

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
        contacts (array): the relative connectivity between different mixing pools (can be float or Dist)

    **Example**::

        import starsim as ss
        mps = ss.MixingPools(
            diseases = 'sis',
            beta = 0.1,
            src = {'0-15': ss.AgeGroup(0, 15), '15+': ss.AgeGroup(15, None)},
            dst = {'0-15': ss.AgeGroup(0, 15), '15+': ss.AgeGroup(15, None)},
            contacts = [[2.4, 0.49], [0.91, 0.16]],
        )
        sim = ss.Sim(diseases='sis', networks=mps).run()
        sim.plot()
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            diseases = None,
            src = None,
            dst = None,
            beta = 0.2,
            contacts = None,
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
            raise TypeError(f'dst must be a provided as a dictionary, not {type(self.pars.src)}')
        p.src = sc.objdict(p.src)
        p.dst = sc.objdict(p.dst)

        # Validate the contacts
        if p.contacts is None:
            p.contacts = np.ones((len(p.src), len(p.dst)))
        p.contacts = np.array(p.contacts)
        actual = p.contacts.shape
        expected = (len(p.src), len(p.dst))
        if actual != expected:
            errormsg = f'The number of source and destination groups must match the number of rows and columns in the mixing matrix, but {actual} != {expected}.'
            raise ValueError(errormsg)

        return

    def init_pre(self, sim):
        super().init_pre(sim)
        p = self.pars

        self.pools = []
        for i,sk,src in p.src.enumitems():
            for j,dk,dst in p.dst.enumitems():
                contacts = p.contacts[i,j]
                if sc.isnumber(contacts): # If it's a number, convert to a distribution
                    contacts = ss.poisson(lam=contacts)
                name = f'pool:{sk}->{dk}'
                mp = MixingPool(name=name, diseases=p.diseases, beta=p.beta, contacts=contacts, src=src, dst=dst)
                mp.init_pre(sim) # Initialize the pool
                self.pools.append(mp)
        return

    def init_post(self):
        """ Initialize each mixing pool """
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
        return


class MixingPool(Route):
    """
    Define a single mixing pool; can be used as a drop-in replacement for a network.

    Args:
        diseases (str): the diseases that transmit via this mixing pool
        src (inds): source agents; can be AgeGroup(), ss.uids(), or lambda(sim); None indicates all alive agents
        dst (inds): destination agents; as above
        beta (float): overall transmission (note: use a float, not a TimePar; the time component is usually handled by the disease beta)
        contacts (Dist): the number of effective contacts of the destination agents

    **Example**::

        import starsim as ss

        # Set the parameters
        mp_pars = dict(
            src = lambda sim: sim.people.male, # only males are infectious
            dst = None, # all agents are susceptible
            beta = 0.2,
            contacts = ss.poisson(lam=4),
        )

        # Seed 5% of the male population
        def p_init(self, sim, uids):
            return 0.05*sim.people.male

        # Create and run the sim
        sis = ss.SIS(init_prev=p_init)
        mp = ss.MixingPool(mp_pars)
        sim = ss.Sim(diseases=sis, networks=mp)
        sim.run()
        sim.plot()
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            diseases = None,
            src = None,
            dst = None, # Same as src
            beta = 0.2,
            contacts = ss.poisson(1.0),
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.FloatArr('eff_contacts', default=self.pars.contacts, label='Effective number of contacts')
        )

        self.pars.diseases = sc.promotetolist(self.pars.diseases)
        self.diseases = None
        self.src_uids = None
        self.dst_uids = None

        self.p_acquire = ss.bernoulli(p=0) # Placeholder value

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
        raise Exception('src must be either a callable function, e.g. lambda sim: ss.uids(sim.people.age<5), or an array of uids.')

    def remove_uids(self, uids):
        """ If UIDs are supplied explicitly, remove them if people die """
        for key in ['src', 'dst']:
            inds = self.pars[key]
            if isinstance(inds, ss.uids):
                self.pars[key] = inds.remove(uids)
        return

    def compute_transmission(self, rel_sus, rel_trans, disease_beta):
        """
        Calculate transmission

        This is called from Infection.infect() together with network transmission.

        Args:
            rel_sus (float): Relative susceptibility
            rel_trans (float): Relative infectiousness
            disease_beta (float): The beta value for the disease. This is typically calculated as a
                pair of values as networks are bidirectional, however, only the first value
                is used because mixing pools are unidirectional.
        Returns:
            UIDs of agents who acquired the disease at this step
        """
        if disease_beta[0] == 0:
            return []

        # Determine the mixing pool beta value
        beta = self.pars.beta
        if isinstance(beta, ss.beta):
            ss.warn(f'In mixing pools, beta should typically be a float, not {beta}; ignoring time value')
            beta = beta.values
        if sc.isnumber(beta) and beta == 0:
            return []

        # Get source and target UIDs
        self.src_uids = self.get_uids(self.pars.src)
        self.dst_uids = self.get_uids(self.pars.dst)
        if len(self.src_uids) == 0 or len(self.dst_uids) == 0:
            return []

        # Calculate transmission
        trans = np.mean(rel_trans[self.src_uids])
        acq = self.eff_contacts[self.dst_uids] * rel_sus[self.dst_uids]
        p = beta*disease_beta[0]*trans*acq
        self.p_acquire.set(p=p)
        return self.p_acquire.filter(self.dst_uids)

    def step(self):
        return
