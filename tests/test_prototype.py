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



#######

# TODO - should the module be stateful or stateless?

#### GONORRHEA SIMULATION

# people = make_people(100)
# pars = defaultdict(dict)
# pars['gonorrhea']['beta'] = {'random':0.3,'msm':0.5}
# sim = cs.Sim(people, [cs.Gonorrhea], pars=pars)
# sim.run()
# plt.figure()
# plt.plot(sim.tvec, sim.results.gonorrhea.n_infected)

#### HIV SIMULATION

people = make_people(100)
pars = defaultdict(dict)
pars['hiv']['beta'] = {'random':0.3,'msm':0.5}
sim = cs.Sim(people, [cs.HIV], pars=pars, analyzers=cs.CD4_analyzer())
sim.run()
plt.figure()
plt.plot(sim.tvec, sim.results.hiv.n_infected)
plt.title('HIV number of infections')
plt.figure()
plt.plot(sim.tvec,sim.analyzers[0].cd4)
plt.title('CD4 counts')

people = make_people(100)
sim = cs.Sim(people, [cs.HIV], pars=pars, interventions=cs.ART([10,20],[20,40]),analyzers=cs.CD4_analyzer())
sim.run()
plt.figure()
plt.plot(sim.tvec,sim.analyzers[0].cd4)
plt.title('CD4 counts (ART)')

#
#
# # # Custom module by user
# class Gonorrhea_DR(cs.Gonorrhea):
#     default_pars = sc.dcp(cs.Gonorrhea.default_pars)
#     default_pars['p_death'] = 0.3

    # We need to make it so that infection with Gonorrhea and Gonorrhea_DR is mutually exclusive


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