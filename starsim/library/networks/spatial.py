"""
Spatial networks
"""
import numpy as np
import starsim as ss
ss_float_ = ss.dtypes.float

__all__ = ['DiskNet']

class DiskNet(ss.Network):
    """
    Disk graph in which edges are made between agents located within a user-defined radius.

    Interactions take place within a square with edge length of 1. Agents are
    initialized to have a random position and orientation within this square. On
    each time step, agents advance v*dt in the direction they are pointed. When
    encountering a wall, agents are reflected.

    Edges are formed between two agents if they are within r distance of each other.
    """
    def __init__(self, key_dict=None, pars=None, **kwargs):
        """ Initialize """
        super().__init__(key_dict=key_dict)
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