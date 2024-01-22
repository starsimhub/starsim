'''
Define connections between disease modules
'''

import starsim as ss


class Connector(ss.Module):
    def __init__(self, pars=None, diseases=None, *args, **kwargs):
        super().__init__(pars=pars, requires=diseases, *args, **kwargs)
        return


# %% Individual connectors

class simple_hiv_ng(Connector):
    """ Simple connector whereby rel_sus to NG doubles if CD4 count is <200"""

    def __init__(self, pars=None):
        super().__init__(pars=pars, label='HIV-Gonorrhea', diseases=[ss.HIV, ss.Gonorrhea])
        self.pars = ss.omerge({
            'rel_trans_hiv': 2,
            'rel_trans_aids': 5,
            'rel_sus_hiv': 2,
            'rel_sus_aids': 5,
        }, self.pars)
        return

    def update(self, sim):
        """ Specify how HIV increases NG rel_sus and rel_trans """

        sim.people.gonorrhea.rel_sus[sim.people.hiv.cd4 < 500] = self.pars.rel_sus_hiv
        sim.people.gonorrhea.rel_sus[sim.people.hiv.cd4 < 200] = self.pars.rel_sus_aids

        sim.people.gonorrhea.rel_trans[sim.people.hiv.cd4 < 500] = self.pars.rel_trans_hiv
        sim.people.gonorrhea.rel_trans[sim.people.hiv.cd4 < 200] = self.pars.rel_trans_aids

        return