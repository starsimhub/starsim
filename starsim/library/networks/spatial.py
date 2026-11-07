"""
Spatial networks, in which contacts are determined by where agents are.

These exist for models in which distance matters. They run as agents only. For contacts across real places, stratify by place (`sir.stratify(patch=ss.levels(ss.geo.geography('Kenya'), mobility=ss.geo.gravity(away=0.1)))`): an `ss.RandomNet` then pairs people within each place, with a share of their contacts in other places, as a compartmental run does.

Used with: agents and compartments; declared and coded modules
"""
import numpy as np
import starsim as ss
from .theoretical import StaticNet

__all__ = ['DiskNet', 'GridNet']

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
        v (rate):   speed at which agents move, a distance per unit time (e.g. `ss.freqperday(0.05)`)
        pars (dict): the same parameters, as a dict

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
            v = ss.freqperday(0.05), # Velocity
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
        vdt = self.pars.v.to_events(self.t.dt) # The distance moved in the step
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
        self.edges['weight'] = np.ones(len(self.p1), dtype=ss.dtypes.float)

        return


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
        pars (dict): the same parameters, as a dict

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(n_agents=2500, diseases='sir', networks=ssl.GridNet(periodic=True))
        sim.run()
        ```
    """
    own_graph = True

    def __init__(self, diagonal=False, periodic=False, pars=None, **kwargs):
        super().__init__(graph=lattice, pars=pars, diagonal=diagonal, periodic=periodic, **kwargs)
        return

    @staticmethod
    def graph_pars():
        return dict(diagonal=False, periodic=False)
