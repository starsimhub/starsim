"""
Theoretical networks, useful for comparison against analytic results and for debugging.

A compartmental run can use `StaticNet` only by its mean number of contacts; the others run as agents only.

Used with: agents and compartments; declared and coded modules
"""
import inspect
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['ErdosRenyiNet', 'NullNet', 'StaticNet', 'SmallWorldNet', 'ScaleFreeNet']

ss_float = ss.dtypes.float
_ = None # Sentinel for "not provided" default arguments

class ErdosRenyiNet(ss.DynamicNetwork):
    """
    In the Erdos-Renyi network, every possible edge has a probability, p, of
    being created on each time step.

    The degree of each node will have a binomial distribution, considering each
    of the N-1 possible edges connecting this node to the others will be created
    with probability p.

    Please be careful with the `dur` parameter. When set to 0, new edges will be
    created on each time step. If positive, edges will persist for `dur` years.
    Note that the existence of edges from previous time steps will not prevent
    or otherwise alter the creation of new edges on each time step, edges will
    accumulate over time.

    Warning: this network is quite slow compared to `ss.RandomNet`.

    Note: CRN safe.

    Args:
        p (float):      probability that each possible edge is created per timestep
        dur (dur/Dist): how long edges persist; 0 means new edges each timestep
        pars (dict): the same parameters, as a dict

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases='sis', networks=ssl.ErdosRenyiNet(p=0.01))
        sim.run()
        ```
    """
    def __init__(self, pars=None, **kwargs):
        """ Initialize """
        super().__init__()
        self.define_pars(
            p = 0.1, # Probability of each edge
            dur = 0, # Duration of zero ensures that new random edges are formed on each time step
        )
        self.update_pars(pars, **kwargs)
        if sc.isnumber(self.pars.dur): self.pars.dur = ss.years(self.pars.dur) # Interpret numbers as years (the default isn't ss.years(0) so a Dist is allowed)
        self.rng_ints = ss.randint(low=0, high=np.iinfo('uint64').max, dtype=np.uint64) # Used to draw a random number for each agent as part of creating edges; must be uint64 for ss.distributions.combine_rands()
        return

    def add_pairs(self):
        """ Generate contacts """
        people = self.sim.people
        born_uids = (people.age > 0).uids

        # Sample integers
        ints = self.rng_ints.rvs(born_uids)

        # All possible edges are upper triangle of complete matrix
        idx1, idx2 = np.triu_indices(n=len(born_uids), k=1)

        # Use integers to create random numbers per edge: combining one integer per agent is
        # far cheaper than drawing a random number for each of the O(N^2) possible edges
        i1 = ints[idx1]
        i2 = ints[idx2]
        r = ss.distributions.combine_rands(i1, i2) # Uniform on [0, 1]
        edge = r <= self.pars.p

        p1 = born_uids[idx1[edge]]
        p2 = born_uids[idx2[edge]]
        beta = np.ones(len(p1), dtype=ss.dtypes.float)

        if isinstance(self.pars.dur, ss.Dist):
            dur = self.pars.dur.rvs(p1)
        else:
            dur = np.ones(len(p1))*(self.pars.dur/self.t.dt)

        self.append(p1=p1, p2=p2, beta=beta, dur=dur)
        return


class NullNet(ss.Network):
    """
    A convenience class for a network of size n that only has self-connections with a weight of 0.
    This network can be useful for debugging purposes or as a placeholder network during development
    for conditions that require more complex network mechanisms.

    Guarantees there's one (1) contact per agent (themselves), and that their connection weight is zero.

    For an empty network (i.e., no edges), use
    `ss.RandomNet(n_contacts=0)`.

    Note: CRN safe.

    Args:
        n_people (int): number of agents in the network; defaults to the sim's `n_agents`
    """

    def __init__(self, n_people=None, **kwargs):
        self.n = n_people
        super().__init__(**kwargs)
        return

    def init_pre(self, sim):
        """ Initialize the network, with the number of people from the sim if not given

        Args:
            sim (Sim): the sim being initialized
        """
        super().init_pre(sim)
        popsize = sim.pars['n_agents']
        if self.n is None:
            self.n = popsize
        else:
            if self.n > popsize:
                errormsg = f'Please ensure the size of the network ({self.n}) is less than or equal to the population size ({popsize}).'
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


class StaticNet(ss.Network):
    """
    A network class of static partnerships converted from a networkx graph. There's no formation of new partnerships
    and initialized partnerships only end when one of the partners dies. The networkx graph can be created outside Starsim
    if population size is known. Or the graph can be created by passing a networkx generator function to Starsim.

    If "seed=True" is passed as a keyword argument or a parameter in pars, it is replaced with the built-in RNG.
    The parameter "n" is supplied automatically to be equal to n_agents. With no graph, it is a random (Erdős–Rényi)
    graph: the one `nx.fast_gnp_random_graph()` would make, edge for edge, made without NetworkX (see `random_graph()`).

    Note: CRN safe.

    Examples:
        ```python
        # Generate a networkx graph and pass to Starsim
        import networkx as nx
        import starsim as ss
        import starsim.library as ssl
        g = nx.scale_free_graph(n=10000)
        ssl.StaticNet(graph=g)

        # Pass a networkx graph generator to Starsim
        ssl.StaticNet(graph=nx.erdos_renyi_graph, p=0.0001, seed=True)
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
        """ Initialize the network, with a random (Erdős–Rényi) graph if none is given

        Args:
            sim (Sim): the sim being initialized
        """
        super().init_pre(sim)
        self.n_agents = sim.pars.n_agents
        n_contacts = self.pars.pop('n_contacts') # Remove from pars dict, but use only if p is not supplied
        if self.pars.p is None: # Convert from n_contacts to probability
            self.pars.p = n_contacts/self.n_agents
        return

    def mean_contacts(self, sim):
        """ The mean degree of its graph (see `ss.Network.mean_contacts()`), or for a random graph not yet made, its number of contacts or its chance of each pair

        Args:
            sim (Sim): the sim being initialized
        """
        graph = self.nx_graph
        if graph is not None and not callable(graph):
            return 2*graph.number_of_edges()/max(graph.number_of_nodes(), 1), 'the mean degree of its graph'
        elif graph is None or getattr(graph, '__name__', None) in ['fast_gnp_random_graph', 'gnp_random_graph', 'erdos_renyi_graph']: # A random graph: a number of contacts, or a chance of each pair
            p = self.pars.get('p')
            return (ss.solvers.Mixing.mean(self.pars.get('n_contacts'), sim.unit_base), 'its mean number of contacts') if p is None else (p*sim.pars.total_pop, 'its chance of each pair, times the population')
        return None, f' (it is made by {getattr(graph, "__name__", "a function")}, whose mean degree is not known before it runs)'

    def init_post(self):
        super().init_post()
        if 'seed' in self.pars and self.pars.seed is True:
            self.pars.seed = self.dist.rng
        if self.nx_graph is None: # A random (Erdős–Rényi) graph
            if isinstance(self.pars.seed, np.random.Generator) and 0 < self.pars.p < 1 and not self.graph_kwargs:
                p1, p2 = random_graph(self.n_agents, self.pars.p, self.pars.seed)
                if not len(p1):
                    raise ValueError(f'The random graph of {self.n_agents} agents with p={self.pars.p} has no edges')
                self.append(p1=p1, p2=p2, beta=np.ones(len(p1)))
                return
            import networkx as nx # Lazy import since slow
            self.nx_graph = nx.fast_gnp_random_graph
        if callable(self.nx_graph):
            args = inspect.signature(self.nx_graph).parameters
            kw = {k:v for k,v in self.pars.items() if k in args} # Only pass p and seed if the generator accepts them
            try:
                self.nx_graph = self.nx_graph(n=self.n_agents, **kw, **self.graph_kwargs)
            except TypeError as e:
                ss.warn(f"{e}: the graph generator {self.nx_graph.__name__} is not supported: make the graph first, and pass it as graph=")
                raise e
        self.validate_pop(self.n_agents)
        self.get_edges()
        return

    def validate_pop(self, n_agents):
        """ Check that the graph fits the number of agents and has edges

        Args:
            n_agents (int): the number of agents
        """
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


def random_graph(n, p, rng):
    """
    The edges of a random (Erdős–Rényi) graph of n nodes, each pair joined with probability p: the edges, in order, that
    `networkx.fast_gnp_random_graph(n, p, seed=rng)` makes from the same random numbers, without making the graph (many times faster)

    Like it, it skips from one edge to the next among the pairs (v, w) with w < v, in order, by a geometric number of pairs.
    """
    total = n*(n - 1)//2 # The pairs, in order: (1,0), (2,0), (2,1), (3,0), ...
    lp = np.log(1.0 - p)
    pos = np.empty(0, dtype=np.int64) # The place of each edge among the pairs
    last = -1
    while last < total: # Each draw is the skip to the next edge, until one is past the last pair
        skips = (np.log(1.0 - rng.random(int(1.02*(total - last)*p) + 100))/lp).astype(np.int64) + 1
        more = last + np.cumsum(skips)
        pos = np.concatenate([pos, more])
        last = more[-1]
    pos = pos[pos < total]
    v = ((1 + np.sqrt(1 + 8*pos.astype(np.float64)))//2).astype(np.int64) # The row of each place, corrected for rounding
    v -= v*(v - 1)//2 > pos
    v += (v + 1)*v//2 <= pos
    w = pos - v*(v - 1)//2
    order = np.argsort(w, kind='stable') # As networkx lists them: by the lower node, then the higher
    return w[order], v[order]


def small_world(n, k, p, seed):
    """ A Watts-Strogatz graph (see `ssl.SmallWorldNet`) """
    import networkx as nx # Lazy import since slow
    return nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)


def scale_free(n, m, seed):
    """ A Barabási-Albert graph (see `ssl.ScaleFreeNet`) """
    import networkx as nx # Lazy import since slow
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)


class SmallWorldNet(StaticNet):
    """
    A small-world (Watts-Strogatz) network: mostly local contacts, with a few long-range ones

    Agents are on a ring, in contact with their `k` nearest neighbors, and each contact is
    rewired to a random agent with probability `p`. The network is fixed. Note: CRN safe.

    Args:
        k (int): the number of neighbors of each agent
        p (float): the probability of rewiring each contact
        pars (dict): the same parameters, as a dict
    """
    def __init__(self, k=4, p=0.1, pars=None, **kwargs):
        super().__init__(graph=small_world, pars=pars, p=p, k=k, **kwargs)
        return


class ScaleFreeNet(StaticNet):
    """
    A scale-free (Barabási-Albert) network: a few agents have many contacts, and most have few

    Each agent in turn joins `m` agents, with those who already have many contacts being
    more likely to be chosen. The network is fixed. Note: CRN safe.

    Args:
        m (int): the number of contacts each new agent makes
        pars (dict): the same parameters, as a dict
    """
    def __init__(self, m=2, pars=None, **kwargs):
        super().__init__(graph=scale_free, pars=pars, m=m, **kwargs)
        return
