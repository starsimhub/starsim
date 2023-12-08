'''
Define connections between disease modules
'''

import stisim as ss


class Connector(ss.Module):
    def __init__(self, pars=None, diseases=None, *args, **kwargs):
        self.pars = ss.omerge(pars)
        self.results = ss.ndict()
        self.diseases = ss.ndict(diseases)
        return


# %% Individual connectors

class simple_hiv_ng(Connector):
    """ Simple connector whereby rel_sus to NG doubles if CD4 count is <200"""

    def __init__(self, pars=None):
        super().__init__(pars=pars)
        self.pars = ss.omerge({
            'rel_trans_hiv': 2,
            'rel_trans_aids': 5,
            'rel_sus_hiv': 2,
            'rel_sus_aids': 5,
        }, self.pars)  # Could add pars here
        self.diseases = ['hiv', 'gonorrhea']
        return

    def initialize(self, sim):
        # Make sure the sim has the modules that this connector deals with
        # TODO replace this placeholder code with something robust.
        if not set(self.diseases).issubset(sim.diseases.keys()):
            errormsg = f'Missing required diseases {set(self.diseases).difference(sim.diseases.keys())}'
            raise ValueError(errormsg)
        return

    def update(self, sim):
        """ Specify how HIV increases NG rel_sus and rel_trans """

        sim.people.gonorrhea.rel_sus[sim.people.hiv.cd4 < 500] = self.pars.rel_sus_hiv
        sim.people.gonorrhea.rel_sus[sim.people.hiv.cd4 < 200] = self.pars.rel_sus_aids

        sim.people.gonorrhea.rel_trans[sim.people.hiv.cd4 < 500] = self.pars.rel_trans_hiv
        sim.people.gonorrhea.rel_trans[sim.people.hiv.cd4 < 200] = self.pars.rel_trans_aids

        return

class simple_hiv_syph(Connector):
    """ Simple connector whereby rel_sus of HIV doubles if individual has primary syphilis"""

    def __init__(self, pars=None):
        super().__init__(pars=pars)
        self.pars = ss.omerge({
            'rel_sus_hiv': 2.67,

        }, self.pars)  # Could add pars here
        self.diseases = ['hiv', 'syphilis']
        return

    def initialize(self, sim):
        # Make sure the sim has the modules that this connector deals with
        # TODO replace this placeholder code with something robust.
        if not set(self.diseases).issubset(sim.diseases.keys()):
            errormsg = f'Missing required diseases {set(self.diseases).difference(sim.diseases.keys())}'
            raise ValueError(errormsg)
        return

    def update(self, sim):
        """ Specify how syphilis increases HIV rel_trans """

        sim.people.hiv.rel_sus[sim.people.syphilis.primary] = self.pars.rel_sus_hiv

        return