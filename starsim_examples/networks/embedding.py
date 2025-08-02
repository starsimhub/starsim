"""
CRN-safe embedding networks
"""
import numpy as np
import starsim as ss

__all__ = ['EmbeddingNet']

class EmbeddingNet(ss.MFNet):
    """
    Heterosexual age-assortative network based on a one-dimensional embedding.

    Warning: this network is random-number safe, but is very slow compared to
    RandomNet.
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
        import scipy.optimize as spo
        import scipy.spatial as spsp

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