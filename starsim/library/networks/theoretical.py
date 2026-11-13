"""
Theoretical networks, useful for comparison against analytic results and for debugging.

A compartmental run can use `StaticNet` only by its mean number of contacts; the others run as agents only. For a random graph that changes every step, use `ss.RandomNet`; for a fixed one, `ssl.StaticNet(n_contacts=...)` (an Erdős–Rényi graph).

Used with: agents and compartments; declared and coded modules
"""
import inspect
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['NullNet', 'StaticNet', 'SmallWorldNet', 'ScaleFreeNet']

ss_float = ss.dtypes.float
_ = None # Sentinel for "not provided" default arguments

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
        self.append(dict(p1=indices, p2=indices, weight=np.zeros_like(indices)))
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

    Args:
        graph (nx.Graph/func): a NetworkX graph, or a function that makes one from the number of agents `n` (and `p` and `seed`, if it takes them); with none, a random (Erdős–Rényi) graph
        n_contacts (float): for a random graph, the mean number of contacts of each agent (used unless `p` is given)
        p (float): for a random graph, the probability that each pair of agents is in contact
        seed (bool/int): the seed of a graph function; True for the network's own random numbers
        kwargs (dict): other arguments, passed to the graph function

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

    own_graph = False # Whether a subclass makes its own graph from its own parameters (e.g. ssl.SmallWorldNet), so takes no others

    def __init__(self, graph=None, pars=None, **kwargs):
        super().__init__()
        self.nx_graph = graph # Not self.graph, which is the graph of states and transitions of a declared model
        self.define_pars(**self.graph_pars())
        self.graph_kwargs = {} if self.own_graph else {k:kwargs.pop(k) for k in list(kwargs) if k not in self.pars and k not in ss.modules.module_args} # Other arguments are passed to the graph generator
        self.update_pars(pars, **kwargs)
        self.dist = ss.Dist(name='StaticNet')
        return

    @staticmethod
    def graph_pars():
        """ Its parameters, and their defaults: a subclass that makes its own graph has its own """
        return dict(seed=True, p=None, n_contacts=10)

    def init_pre(self, sim):
        """ Initialize the network, with a random (Erdős–Rényi) graph if none is given

        Args:
            sim (Sim): the sim being initialized
        """
        super().init_pre(sim)
        self.n_agents = sim.pars.n_agents
        if not self.own_graph:
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
        if self.own_graph:
            return super().mean_contacts(sim)
        elif graph is not None and not callable(graph):
            return 2*graph.number_of_edges()/max(graph.number_of_nodes(), 1), 'the mean degree of its graph'
        elif graph is None or getattr(graph, '__name__', None) in ['fast_gnp_random_graph', 'gnp_random_graph', 'erdos_renyi_graph']: # A random graph: a number of contacts, or a chance of each pair
            p = self.pars.get('p')
            return (ss.solvers.Mixing.mean(self.pars.get('n_contacts'), sim.unit_base), 'its mean number of contacts') if p is None else (p*sim.pars.total_pop, 'its chance of each pair, times the population')
        return None, f' (it is made by {getattr(graph, "__name__", "a function")}, whose mean degree is not known before it runs)'

    def init_post(self):
        super().init_post()
        if 'seed' in self.pars and self.pars.seed is True:
            self.pars.seed = self.dist.rng
        seed = self.pars.seed if 'seed' in self.pars else self.dist.rng # A subclass's graph uses the network's own random numbers
        if self.nx_graph is None: # A random (Erdős–Rényi) graph
            if isinstance(self.pars.seed, np.random.Generator) and 0 < self.pars.p < 1 and not self.graph_kwargs:
                p1, p2 = random_graph(self.n_agents, self.pars.p, self.pars.seed)
                if not len(p1):
                    raise ValueError(f'The random graph of {self.n_agents} agents with p={self.pars.p} has no edges')
                self.append(p1=p1, p2=p2, weight=np.ones(len(p1)))
                return
            import networkx as nx # Lazy import since slow
            self.nx_graph = nx.fast_gnp_random_graph
        if callable(self.nx_graph):
            args = inspect.signature(self.nx_graph).parameters
            kw = {k:v for k,v in dict(self.pars, seed=seed).items() if k in args} # Only pass p and seed if the generator accepts them
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
        edges = dict(p1=p1s, p2=p2s, weight=np.ones_like(p1s))
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


def small_world(n, n_contacts, rewire, seed):
    """ A Watts-Strogatz graph (see `ssl.SmallWorldNet`) """
    import networkx as nx # Lazy import since slow
    return nx.watts_strogatz_graph(n=n, k=n_contacts, p=rewire, seed=seed)


def scale_free(n, n_contacts, seed):
    """ A Barabási-Albert graph (see `ssl.ScaleFreeNet`), in which each new node joins half as many as the mean number of contacts """
    import networkx as nx # Lazy import since slow
    if n_contacts % 2 or n_contacts < 2:
        errormsg = f'ssl.ScaleFreeNet(n_contacts={n_contacts}): each agent in turn joins half that many others, so it is a whole even number, at least 2'
        raise ValueError(errormsg)
    return nx.barabasi_albert_graph(n=n, m=int(n_contacts)//2, seed=seed)


class SmallWorldNet(StaticNet):
    """
    A small-world (Watts-Strogatz) network: mostly local contacts, with a few long-range ones

    Agents are on a ring, in contact with their `n_contacts` nearest neighbors, and each contact is
    rewired to a random agent with probability `rewire`. The network is fixed. Note: CRN safe.

    Args:
        n_contacts (int): the number of neighbors of each agent (even; an odd number is rounded down)
        rewire (float): the probability of rewiring each contact
        pars (dict): the same parameters, as a dict
    """
    own_graph = True

    def __init__(self, n_contacts=4, rewire=0.1, pars=None, **kwargs):
        super().__init__(graph=small_world, pars=pars, n_contacts=n_contacts, rewire=rewire, **kwargs)
        return

    @staticmethod
    def graph_pars():
        return dict(n_contacts=4, rewire=0.1)


class ScaleFreeNet(StaticNet):
    """
    A scale-free (Barabási-Albert) network: a few agents have many contacts, and most have few

    Each agent in turn joins half of `n_contacts` agents, with those who already have many contacts being
    more likely to be chosen, so the mean number of contacts is `n_contacts`. The network is fixed. Note: CRN safe.

    Args:
        n_contacts (int): the mean number of contacts of each agent (a whole even number)
        pars (dict): the same parameters, as a dict
    """
    own_graph = True

    def __init__(self, n_contacts=4, pars=None, **kwargs):
        super().__init__(graph=scale_free, pars=pars, n_contacts=n_contacts, **kwargs)
        return

    @staticmethod
    def graph_pars():
        return dict(n_contacts=4)
