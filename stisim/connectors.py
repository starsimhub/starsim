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
        self.pars = ss.omerge({}, self.pars)  # Unnecessary but could add pars here
        self.modules = [ss.HIV, ss.Gonorrhea]
        return

    def initialize(self, sim):
        # Make sure the sim has the modules that this connector deals with
        if (ss.HIV() not in sim.modules) or (ss.Gonorrhea not in sim.modules):
            errormsg = 'Missing required modules'
            raise ValueError(errormsg)
        return

    def apply(self, sim):
        """ Specify how HIV increases NG rel_sus and rel_trans """

        sim.people.gonorrhea.rel_sus[sim.people.hiv.cd4 < 500] *= 2
        sim.people.gonorrhea.rel_sus[sim.people.hiv.cd4 < 200] *= 5

        sim.people.gonorrhea.rel_trans[sim.people.hiv.cd4 < 500] *= 2
        sim.people.gonorrhea.rel_trans[sim.people.hiv.cd4 < 200] *= 5

        return




