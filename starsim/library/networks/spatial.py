"""
Spatial networks, in which contacts are determined by agents' positions.
"""
import numpy as np
import starsim as ss

class DiskNet(ss.Network):
    """
    Disk graph in which edges are made between agents located within a user-defined radius.

    Interactions take place within a square with edge length of 1. Agents are
    initialized to have a random position and orientation within this square. On
    each time step, agents advance v*dt in the direction they are pointed. When
    encountering a wall, agents are reflected.

    Edges are formed between two agents if they are within r distance of each other.

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
            v = ss.freq(0.05, unit=ss.day), # Velocity
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