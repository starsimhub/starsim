import sciris as sc
import numpy as np
from .people import make_people

class Sim():

    default_pars = sc.objdict(
        n=1000,
        npts=30,
        dt=1
    )

    def __init__(self, people=None, modules=None, pars=None, interventions=None, analyzers=None):
        # TODO - clearer options for time units?
        #   ti - index self.ti = 0
        #    t - floating point year e.g., 2020.5
        #   dt - simulation primary step size (years per step)  # TODO - use-case for model-specific dt?
        # date - '20200601'

        self.ti = 0

        self.pars = sc.dcp(self.default_pars)
        if pars is not None:
            self.pars.update(pars)
        self.people = people if people is not None else make_people(self.pars['n'])
        self.modules = sc.promotetolist(modules)
        self.interventions = sc.promotetolist(interventions)
        self.analyzers = sc.promotetolist(analyzers)
        self.results = sc.objdict()

    @property
    def dt(self):
        return self.pars['dt']

    @property
    def t(self):
        return self.ti*self.dt

    @property
    def n(self):
        return len(self.people)

    @property
    def tvec(self):
        return np.arange(0,self.npts)*self.dt

    @property
    def npts(self):
        return self.pars['npts']

    def initialize(self):
        self.people.initialize(self)

        for module in self.modules:
            module.initialize(self)

        for intervention in self.interventions:
            intervention.initialize(self)

        for analyzer in self.analyzers:
            analyzer.initialize(self)


    def run(self):
        self.initialize()
        for i in range(self.pars.npts):
            self.step()
        self.people.finalize_results(self)

        for module in self.modules:
            module.finalize_results(self)

    def step(self):

        self.people.update_states(self)

        for intervention in self.interventions:
            intervention.apply(self)

        for module in self.modules:
            module.make_new_cases(self)

        self.people.update_results(self)

        for analyzer in self.analyzers:
            analyzer.apply(self)

        self.ti += self.pars.dt
