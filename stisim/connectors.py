'''
Define connections between disease modules
'''

import stisim as ss


class Connector(ss.Module):
    def __init__(self, pars=None, modules=None, *args, **kwargs):
        self.pars = ss.omerge(pars)
        self.results = ss.ndict()
        self.modules = ss.ndict(modules)
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
        self.modules = ['hiv', 'gonorrhea']
        return

    def initialize(self, sim):
        # Make sure the sim has the modules that this connector deals with
        # TODO replace this placeholder code with something robust.
        if not set(self.modules).issubset(sim.modules.keys()):
            errormsg = f'Missing required modules {set(self.modules).difference(sim.modules.keys())}'
            raise ValueError(errormsg)
        return

    def update(self, sim):
        """ Specify how HIV increases NG rel_sus and rel_trans """

        sim.people.gonorrhea.rel_sus[sim.people.hiv.cd4 < 500] = self.pars.rel_sus_hiv
        sim.people.gonorrhea.rel_sus[sim.people.hiv.cd4 < 200] = self.pars.rel_sus_aids

        sim.people.gonorrhea.rel_trans[sim.people.hiv.cd4 < 500] = self.pars.rel_trans_hiv
        sim.people.gonorrhea.rel_trans[sim.people.hiv.cd4 < 200] = self.pars.rel_trans_aids

        return


class ng_infertility(Connector):
    """
    A connector between the pregnancy and gonorrhea modules, which reduces a woman's
    fertility if she has untreated gonorrhea infection
    """
    pass


class ng_birth(Connector):
    """
    A connector between the pregnancy and gonorrhea modules, which adds adverse birth
    outcomes for babies born to mothers with untreated gonorrhea infection.
    """
    pass


