"""
Additional theoretical network classes
"""
import numpy as np
import starsim as ss
ss_float_ = ss.dtypes.float

__all__ = ['ErdosRenyiNet', 'NullNet']

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
    """
    def __init__(self, key_dict=None, pars=None, **kwargs):
        """ Initialize """
        super().__init__(key_dict=key_dict)
        self.define_pars(
            p = 0.1, # Probability of each edge
            dur = ss.dur(0), # Duration of zero ensures that new random edges are formed on each time step
        )
        self.update_pars(pars, **kwargs)
        self.randint = ss.randint(low=np.iinfo('int64').min, high=np.iinfo('int64').max, dtype=np.int64) # Used to draw a random number for each agent as part of creating edges
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