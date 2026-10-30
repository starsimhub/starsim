"""
Spatial networks, in which contacts are determined by where agents are.
"""
import numpy as np
import scipy.sparse as spsp
import starsim as ss
from .theoretical import StaticNet

_ = None # Sentinel for "not provided" default arguments

class DiskNet(ss.Network):
    """
    Disk graph in which edges are made between agents located within a user-defined radius.

    Interactions take place within a square with edge length of 1. Agents are
    initialized to have a random position and orientation within this square. On
    each time step, agents advance v*dt in the direction they are pointed. When
    encountering a wall, agents are reflected.

    Edges are formed between two agents if they are within r distance of each other.

    Note: CRN safe.

    Args:
        r (float):  radius within which edges are formed
        v (freq):   speed at which agents move

    Attributes:
        x (FloatArr):       x position, in [0, 1]
        y (FloatArr):       y position, in [0, 1]
        theta (FloatArr):   direction of travel, in radians

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases='sis', networks=ssl.DiskNet(r=0.05))
        sim.run()
        ```
    """
    def __init__(self, pars=None, **kwargs):
        """ Initialize """
        super().__init__()
        self.define_pars(
            r = 0.1, # Radius
            v = ss.time.freq(0.05, 'day'), # Velocity
        )
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.FloatArr('x', default=ss.random(), label='X position'),
            ss.FloatArr('y', default=ss.random(), label='Y position'),
            ss.FloatArr('theta', default=ss.uniform(high=2*np.pi), label='Heading'),
        )
        return

    def step(self):
        # Motion step
        vdt = self.pars.v * self.t.dt
        x = (self.x + vdt * np.cos(self.theta)) % 2
        y = (self.y + vdt * np.sin(self.theta)) % 2

        # Wall bounce: after the modulo, positions in (1, 2) have bounced an odd number of times, so reflect them and their heading
        bx = x > 1
        by = y > 1
        self.x[:] = np.where(bx, 2 - x, x)
        self.y[:] = np.where(by, 2 - y, y)
        self.theta[:] = np.where(bx, np.pi - self.theta, self.theta)
        self.theta[:] = np.where(by, -self.theta, self.theta)

        self.add_pairs()
        return

    def add_pairs(self):
        """ Generate contacts """
        uids = self.sim.people.auids
        x = self.x.values
        y = self.y.values
        p1, p2 = np.triu_indices(n=len(uids), k=1)
        d12_sq = (x[p2]-x[p1])**2 + (y[p2]-y[p1])**2
        edge = d12_sq < self.pars.r**2

        self.edges['p1'] = uids[p1[edge]]
        self.edges['p2'] = uids[p2[edge]]
        self.edges['beta'] = np.ones(len(self.p1), dtype=ss.dtypes.float)

        return


class GeoNet(ss.RandomNet):
    """
    Random contacts between real places

    Each agent lives in a place, in proportion to its population. Most of an agent's
    contacts are in their own place, and the rest are in other places, more of them in
    places that are large and near, as in a gravity model: the share going from place i
    to place j is in proportion to `pop_j/distance_ij**c`. Within each place, contacts
    are paired at random. A country's places need no download (see `ss.geo.geography`).

    The place of each agent is the state `place`, an index into `geo.names`. Like
    `ss.RandomNet`, this is not CRN safe when people are born or die.

    Args:
        places (str/`ss.geo.geography`): a country by name, or a geography with populations and distances
        level (int): for a country, the administrative level (see `ss.geo.geography`)
        local (float): the share of contacts in an agent's own place
        c (float): how quickly contacts fall off with distance
        kwargs (dict): passed to `ss.RandomNet`, e.g. `n_contacts` and `dur`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(n_agents=10_000, diseases='sis', networks=ssl.GeoNet('Kenya', n_contacts=10))
        sim.run()
        ```
    """
    def __init__(self, places='Kenya', level=1, local=0.9, c=2.0, pars=None, **kwargs):
        super().__init__(pars, **kwargs)
        self.geo = places if isinstance(places, ss.geo.geography) else ss.geo.geography(places, level=level)
        if self.geo.pop is None or self.geo.distances is None:
            errormsg = f'ssl.GeoNet() needs the populations and distances of its places, which {self.geo} does not have'
            raise ValueError(errormsg)
        self.local = local
        self.c = c
        self.share = self.contact_shares()
        self.define_states(
            ss.IntArr('place', default=ss.choice(a=len(self.geo), p=self.geo.pop/self.geo.pop.sum()), label='Place'),
        )
        return

    def contact_shares(self):
        """ The share of the contacts of the people of each place (rows) that are in each place (columns), as a sparse array """
        pop, dist = self.geo.pop, self.geo.distances
        with np.errstate(divide='ignore'):
            away = pop[None, :]/dist**self.c
        away[~np.isfinite(away)] = 0 # Including each place itself
        totals = away.sum(axis=1, keepdims=True)
        away = np.divide(away, totals, out=np.zeros_like(away), where=totals > 0)
        local = np.where(totals.ravel() > 0, self.local, 1.0) # With nowhere else to go, all are local
        return spsp.csr_array((1 - local)[:, None]*away + np.diag(local))

    def pair(self, inds, n_contacts):
        """ Find edges (see `ss.RandomNet.pair()`): each contact is in a place drawn from the shares of the agent's own place """
        stubs = np.repeat(inds, n_contacts)
        place = self.draw_places(self.share, self.place.raw[stubs], self.dist.rng.random(len(stubs)))
        self.dist.jump()
        return self.pair_within(stubs, place, len(self.geo), inds, n_contacts)


def lattice(n, diagonal=False, periodic=False):
    """ A graph of `n` points on a square grid, numbered row by row (see `ssl.GridNet`) """
    import networkx as nx # Lazy import since slow
    side = int(np.ceil(np.sqrt(n)))
    steps = [(0, 1), (1, 0)] + [(1, 1), (1, -1)]*diagonal
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        row, col = divmod(i, side)
        for dr, dc in steps:
            r, c = row + dr, col + dc
            if periodic:
                r, c = r % side, c % side
            if 0 <= r < side and 0 <= c < side and r*side + c < n and r*side + c != i:
                graph.add_edge(i, r*side + c)
    return graph


class GridNet(StaticNet):
    """
    Agents on a square grid, in contact with their neighbors

    Agents are placed row by row on the smallest square that holds them, as in lattice
    models of spatial spread. The network is fixed, so agents born later have no
    contacts. Note: CRN safe.

    Args:
        diagonal (bool): if True, agents have 8 neighbors rather than 4
        periodic (bool): if True, the edges of the grid wrap around, so that everyone has the same number of neighbors

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(n_agents=2500, diseases='sir', networks=ssl.GridNet(periodic=True))
        sim.run()
        ```
    """
    def __init__(self, diagonal=False, periodic=False, pars=None, **kwargs):
        super().__init__(graph=lattice, pars=pars, diagonal=diagonal, periodic=periodic, **kwargs)
        return
