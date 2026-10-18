"""
Theoretical networks, useful for comparison against analytic results and for debugging.
"""
import numpy as np
import sciris as sc
import starsim as ss

ss_float = ss.dtypes.float
_ = None # Sentinel for "not provided" default arguments

class ErdosRenyiNet(ss.DynamicNetwork):
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

    Warning: this network is quite slow compared to `ss.RandomNet`.

    Args:
        p (float):      probability that each possible edge is created per timestep
        dur (dur/Dist): how long edges persist; 0 means new edges each timestep

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
        self.rng_ints = ss.randint(low=0, high=np.iinfo('uint64').max, dtype=np.uint64) # Used to draw a random number for each agent as part of creating edges; must be uint64 for ss.utils.combine_rands()
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
        r = ss.utils.combine_rands(i1, i2) # Uniform on [0, 1] # TODO: use ss.multi_rand()
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

    For an empty network (ie, no edges) use
    >> import starsim as ss
    >> import networkx as nx
    >> empty_net_static = ss.StaticNet(nx.empty_graph)
    >> empty_net_rand = ss.RandomNet(n_contacts=0)

    Args:
        n_people (int): number of agents in the network; defaults to the sim's `n_agents`
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


class RandomSafeNet(ss.DynamicNetwork):
    """
    Create a CRN-safe, O(N) random network

    This network is similar to `ss.RandomNet(method='exact')`, but is random-number safe
    (i.e., the addition of a single new agent will not perturb the entire rest
    of the network). However, it is somewhat slower than `ss.RandomNet()`,
    so should be used where CRN safety is important (e.g., scenario analysis).

    Note: `ss.RandomNet` uses `n_contacts`, which is the total number of contacts
    per agent. `ssl.RandomSafeNet` uses `n_edges`, which is the total number of
    *edges* per agent. Since contacts are usually bidirectional, n_contacts = 2*n_edges.
    For example, `ss.RandomNet(n_contacts=10, method='exact')` will give (nearly) identical results
    to `ssl.RandomSafeNet(n_edges=5)`. In addition, whereas `n_contacts` can be
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
        if sc.isnumber(self.pars.dur): self.pars.dur = ss.years(self.pars.dur) # Interpret numbers as years, as for ss.RandomNet (the default isn't ss.years(0) so a Dist is allowed)
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
