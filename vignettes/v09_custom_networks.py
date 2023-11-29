"""
Vignette 09: Custom networks
TO RESOLVE:
    - what should the default networks be?
    - how should network participation be specified? e.g. not everyone will participate in the MSM or FSW networks
    - what should happen if network properties depend on disease attributes? (e.g. Romesh's example)
    - what should happen if the network depends on custom people attributes?
"""

import stisim as ss
import numpy as np


##################
# @RomeshA
##################
people = ss.People.from_households(n=1000, <some household structure inputs>)  # Creates a household layer
people = ss.People(n=1000) # No contact layers are present
people.contacts['msm'] = ss.SexualMixingLayer(label='msm', people=people, eligible='m',mean_partners=2,dur=scipy.stats.lognorm(loc=5, s=3))
people.contacts['msm'] = ss.SexualMixingLayer(label='msm', people=people, eligible='m',mean_partners=2,dur=ss.lognormal(mean=5, std=3))

# What if the layer depends on some of the disease attributes?
# e.g.

class CasualMixingLayer(ss.SexualMixingLayer):
    def update(self, sim):
        n_contacts = self.mean_partners*np.ones(sim.n)
        n_contacts[sim.people.gonorrhea.symptomatic] *= 0.5
        super().update(sim, n_contacts)

people = ss.People(n=1000) # No contact layers are present
sim = ss.Sim(modules=[ss.HPV(), ss.Gonorrhea()], people=people)
sim.initialize_modules()
people.contacts['msm'] = ss.CasualMixingLayer(label='casual', sim=sim, eligible='m',mean_partners=2,dur=scipy.stats.lognorm(loc=5, s=3))


##################
# @robynstuart
##################
# Case 1: in-built network options
# Some defaults for networks can be selected via 'network_structure' parameter (akin to 'pop_type' in covasim):
# The default, which is an alias for sim = ss.Sim(networks='mf', modules=...) & equivalent to sim = ss.Sim(modules=...)
sim = ss.Sim(network_structure='random', modules=ss.gonorrhea())
# The 'hybrid' option adds MSM, & is an alias for adding an MSM state and adding the network
sim = ss.Sim(network_structure='hybrid', modules=ss.gonorrhea())
# This is an alias for:
states = ss.StochState('msm', bool, distdict=dict(dist='choice', par1=[1, 0], par2=[0.1, 0.9]))
sim = ss.Sim(states=states, networks=['mf', 'msm'], modules=ss.gonorrhea())

# Each disease should have default beta layer values for each built-in network
# The module should use the beta value for whichever networks have been added, e.g.
syph = ss.syphylis(
    beta=0.0008,  # Overall baseline beta. By convention this should represent M->F transmission risk
    beta_layer=dict(mf=[1, 0.5], msm=[10, 10])
)
sim = ss.Sim(network_structure='hybrid', modules=syph)


# Case 2: adding a custom network
ppl = ss.People(100)
nwk = ss.mf()
syph = ss.syphilis()
sim['syphilis']['beta']['mf'] *= 2  # Increase beta for syph transmission
sim = ss.Sim(people=ppl, networks=nwk, modules=syph)


# Case 3: adding a custom network that needs specific people attributes
is_fsw = ss.StochState('fsw', bool, distdict=dict(dist='choice', par1=[1, 0], par2=[0.05, 0.95]))
is_client = ss.StochState('client', bool, distdict=dict(dist='choice', par1=[1, 0], par2=[0.15, 0.85]))
ppl = ss.People(100, states=[is_fsw, is_client])
sex_work = sex_work()  # Question: what constraints are there on this? Must it be a ss.Network or could it be anything?
syph = ss.syphilis()
sim = ss.Sim(people=ppl, networks=['mf', sex_work], modules=syph)



