'''
Define connections between disease modules
'''

import stisim as ss


class Connector(ss.Module):
    def __init__(self, pars=None, modules=None, *args, **kwargs):
        self.pars = ss.omerge(pars)
        self.states = ss.ndict()
        self.results = ss.ndict()
        self.modules = ss.ndict()
        return


class Connectors(ss.ndict):
    def __init__(self, *args, type=Connector, **kwargs):
        return super().__init__(self, *args, type=type, **kwargs)


#%% Individual connectors

class simple_hiv_ng(Connector):
    """ Simple connector whereby rel_sus to NG doubles if CD4 count is <200"""

    def __init__(self, pars=None):
        super().__init__(pars=pars)
        self.pars = ss.omerge({
            'rel_trans_hiv': 2,
            'rel_trans_aids': 5,
            'rel_sus_hiv': 2,
            'rel_sus_aids': 5,
        }, self.pars)  # Unnecessary but could add pars here
        self.modules = [ss.HIV, ss.Gonorrhea]
        return

    def initialize(self, sim):
        # Make sure the sim has the modules that this connector deals with
        if ('hiv' not in sim.modules.keys()) or ('gonorrhea' not in sim.modules.keys()):
            errormsg = 'Missing required modules'
            raise ValueError(errormsg)
        return

    def update(self, sim):
        """ Specify how HIV increases NG rel_sus and rel_trans """

        sim.people.gonorrhea.rel_sus[sim.people.hiv.cd4 < 500] = self.pars.rel_sus_hiv
        sim.people.gonorrhea.rel_sus[sim.people.hiv.cd4 < 200] = self.pars.rel_sus_aids

        sim.people.gonorrhea.rel_trans[sim.people.hiv.cd4 < 500] = self.pars.rel_trans_hiv
        sim.people.gonorrhea.rel_trans[sim.people.hiv.cd4 < 200] = self.pars.rel_trans_aids

        return




