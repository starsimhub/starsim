"""
CRN-safe embedding networks -- illustrative example only due to poor performance.
"""
import numpy as np
import starsim as ss
import scipy.optimize as spo
import scipy.spatial as spsp

__all__ = ['EmbeddingNet']

def embedding_loc(module, sim, uids):
    """ The mean position of each person in the embedding: their age, shifted for women so they are paired with older men """
    loc = sim.people.age[uids]
    loc[sim.people.female[uids]] += module.pars.male_shift
    return loc


class EmbeddingNet(ss.SexualNet):
    """
    Heterosexual age-assortative network based on a one-dimensional embedding.

    Warning: this network is random-number safe, but is very slow compared to
    RandomNet.
    """

    male_shift = ss.par(5) # The average number of years by which men are older than women in partnerships (the parameters of ss.SexualNet are kept)
    embedding_func = ss.par(ss.normal(name='EmbeddingNet', mean=embedding_loc, std=2)) # Each person's position in the embedding

    def add_pairs(self):
        people = self.sim.people
        available_m = self.available(people, 'male')
        available_f = self.available(people, 'female')

        if not len(available_m) or not len(available_f):
            if ss.options.verbose > 1:
                print('No pairs to add')
            return 0

        available = ss.uids.concatenate(available_m, available_f)
        loc = self.pars.embedding_func.rvs(available)
        loc_f = loc[people.female[available]]
        loc_m = loc[~people.female[available]]

        dist_mat = spsp.distance_matrix(loc_m[:, np.newaxis], loc_f[:, np.newaxis])
        ind_m, ind_f = spo.linear_sum_assignment(dist_mat)
        n_pairs = len(ind_f)

        # Finalize pairs
        p1 = available_m[ind_m]
        p2 = available_f[ind_f]
        weight = np.ones(n_pairs)
        dur_vals = self.pars.dur.rvs(p1)
        act_vals = self.pars.acts.rvs(p1)

        self.append(p1=p1, p2=p2, weight=weight, dur=dur_vals, acts=act_vals)
        return len(weight)