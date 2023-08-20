"""
Experimenting with networks
Key properties of networks:
    - who participates? consider eligibility vs uptake, e.g. all females 15-49 *can* be FSW but not all are
    - how do participants in the network choose their partners?
    - are participants in one network restricted from participating in other networks?

"""
import sciris

import stisim as ss
from scipy.stats import norm

####################################
# Adding states to people
####################################

msm = ss.msm()
ppl = ss.People(100)
ppl.age = norm.rvs(30, 10, 100)  # Give people non-zero ages
ppl.add_network(msm)

import sciris as sc

{'MSM': ['a']}
{'MF': ['m', 'c']}
{'Maternal': ['a']}
{'Respiratory': ['school', 'work']}
{'Social': ['internet', 'behavioral']}

# MSM network
sim.people.contacts['msm']

# people.msm stores MSM specific states, e.g.
# and is a network
sim.people.msm.participant
sim.people.msm.active
sim.people.msm.debut


sim.people.networks.sexual_contacts  # is this also a network??
# this network aggregates all the sub-networks
# it has special methods for excluding double pairs for e.g.
sim.people.networks.injecting  # second type of network for STIsim
sim.people.networks.maternal

# here, sim.people.msm is NOT a network

sim.people.networks.sexual_contacts.msm  # and this?
sim.people.networks.sexual_contacts.participant
sim.people.networks.sexual_contacts.msm.participant

sim.people.is_msm # refers to sim.people.networks.sexual_contacts.participant

class Contacts(ss.Network):
    pass

part_rates = {
    'm': {
        'msm': 0.1,
        'mf_marital': 0.6,
        'mf_casual': 0.1,
        # ('msm', 'mf_marital'): 0.05,
        # ('msm', 'mf_casual'): 0.05,
        # ('mf_marital', 'mf_casual'): 0.05,
        # ('msm', 'mf_marital', 'mf_casual'): 0.05,
    },
    'f': {
        'msm': 0,
        'mf_marital': 0.7,
        'mf_casual': 0.2,
        ('msm', 'mf_marital'): 0,
        ('msm', 'mf_casual'): 0,
        ('mf_marital', 'mf_casual'): 0.1,
        ('msm', 'mf_marital', 'mf_casual'): 0,
    }
}
mixing = Mixing(participation=part_rates)

# Sexually active MSM
# ppl.networks['msm'].active(ppl)
# What about ppl.msm.active? Should ppl.msm be an objdict of states (as it is now), or the network itself?


