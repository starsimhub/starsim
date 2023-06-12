# TOY MODEL
# This toy model has HIV and gonorrhea with coinfection. The diseases interact
# such that gonorrhea susceptibility depends on the HIV CD4 value, and the HIV
# module includes ART as a state which decreases the impact of HIV coinfection by
# raising the CD4 value.

# Some challenges
# - How would you make it so that people infected with gonorrhea reduce sexual activity
#   until their infection is cleared?

import numpy as np
import sciris as sc
from collections import defaultdict
import matplotlib.pyplot as plt
import concept as cs

def make_people(n):
    people = cs.People(n)
    people.contacts['random'] = cs.RandomDynamicSexualLayer(people)
    people.contacts['msm'] = cs.StaticLayer(people,'mm')

    # relationship types, homosexual/heterosexual, births?
    # do we model pregnancies? track parents? households?
    return people


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

        self.people.initialize(sim)
        for module in self.modules:
            module.initialize(sim)


    def run(self):
        self.initialize()
        for i in range(self.pars.npts-1):
            self.step()
        self.people.finalize_results(sim)

        for module in self.modules:
            module.finalize_results(sim)

    def step(self):
        self.ti += self.pars.dt

        self.people.update_states_pre(sim)

        for module in self.modules:
            module.transmit(self)

        self.people.update_results(sim)


#######

# TODO - should the module be stateful or stateless?

people = make_people(100)
pars = defaultdict(dict)
pars['gonorrhea']['beta'] = {'random':0.3,'msm':0.5}
sim = Sim(people, [cs.Gonorrhea], pars=pars)
sim.run()
plt.figure()
plt.plot(sim.tvec, sim.results.gonorrhea.n_infected)


# # Custom module by user
# class Gonorrhea_DR(Gonorrhea):
#     default_pars = sc.dcp(Gonorrhea.default_pars)
#     default_pars['p_death'] = 0.3
#
# How should coinfection with these two modules be prevented?
#
#
# people = make_people(100)
# pars = defaultdict(dict)
# pars['gonorrhea']['beta'] = {'random':0.1,'msm':0.15}
# pars['gonorrhea_dr']['beta'] = {'random':0.05,'msm':0.1}
# sim = Sim(people, [Gonorrhea, Gonorrhea_DR], pars=pars)
# sim.run()
# plt.figure()
# plt.plot(sim.tvec, sim.results.gonorrhea.prevalence)
# plt.plot(sim.tvec, sim.results.gonorrhea_dr.prevalence, color='r')
# plt.show()