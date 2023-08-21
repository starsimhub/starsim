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

####################################
# How to represent MSM status
####################################
people = ss.People(100)
# Old way:
# Networks were stored in a dictionary of networks
# people.contacts['msm']

# New way:
# people.routes.sexual is a single network that aggregates all subnetworks
# We may also have
# people.routes.injecting  # second type of network for STIsim
# people.routes.airborne  # if modeling TB for e.g.

#########################################################
# Construction
#########################################################
import stisim.people as ssppl
import stisim.utils as ssu

def bs(name, prob, eligibility=None):
    """ Shorthand for defining a boolean state with a certain probability """
    distdict = dict(dist='choice', par1=[True, False], par2=[prob, 1 - prob])
    bool_state = ssppl.State(name, bool, distdict=distdict, eligibility=eligibility)
    return bool_state


class Network:
    def __init__(self, pars, eligibility=None, states=None):
        self.pars = ssu.omerge(dict(
            part_rate=1,
        ), pars)
        self.eligibility = eligibility
        self.states = ssu.omerge(
            ssu.named_dict(
                bs('participant', self.pars['part_rate'], eligibility=self.eligibility),
                ssppl.State('debut', float, 15),
                ssppl.State('partners', int, 2),
                ssppl.State('current_partners', int, 0),
        ), states)


class msm(Network):
    def __init__(self, pars, eligibility='male', states=None):
        super().__init__(pars, eligibility, states)
        self.states = ssu.omerge(self.states, ssu.named_dict(states))


class mf_marital(Network):
    def __init__(self, pars, eligibility=None, states=None):
        super().__init__(pars, eligibility, states)


class mf_casual(Network):
    def __init__(self, pars, eligibility=None, states=None):
        super().__init__(pars, eligibility, states)


class SexualTransmission(Network):
    def __init__(self, networks=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.networks = networks

    def initialize(self, people):
        """ Assign membership to each network """
        self.find_participants(people)

    def find_participants(self, people):
        participants = ssu.true(self.states['participant'])

        for network in self.networks:
            network.states['participant'].get_eligibility(people)


# Example
unmarried = lambda ppl: ppl.networks.sexual_contacts.mf_marital.current_partners == 0
married_msm = lambda ppl, rate: ppl.networks.sexual_contacts.mf_marital.current_partners == 0
networks = [
    msm(pars=dict(part_rate=0.1)),
    mf_casual(pars=dict(part_rate=0.3), eligibility=unmarried),  # Participation rates for these layers might be age/sex dependent
    mf_marital(pars=dict(part_rate=0.7))
]
st = SexualTransmission(networks=networks)


# Networks store states
# people.routes.sexual.msm.participant
# people.routes.sexual.msm.active
# people.routes.sexual.msm.debut
# Can use
# people.is_msm as references to people.routes.sexual.msm.participant

# The overall sexual network aggregates all the component networks
# It is also a network instance
# It has special methods for excluding double pairs
#  sim.people.routes.sexual


# # here, sim.people.msm is NOT a network
#
# sim.people.networks.sexual_contacts.msm  # and this?
# sim.people.networks.sexual_contacts.participant
# sim.people.networks.sexual_contacts.msm.participant
#
# sim.people.is_msm # refers to sim.people.networks.sexual_contacts.participant
#
# # Sexually active MSM
# # ppl.networks['msm'].active(ppl)
# # What about ppl.msm.active? Should ppl.msm be an objdict of states (as it is now), or the network itself?
#
# import stisim.utils as ssu
# import stisim.people as ssppl
#
# # Make a school network
# def school_aged(people): return (people.age > 5) & (people.age < 18)
# in_school = ssppl.State(
#     'in_school',
#     bool,
#     fill_value=ssu.sample('choice', par1=[True,False], par2=[0.9,0.1]),
#     eligibility=school_aged
# )
# school_network = ss.school(states=in_school)
#
# # Make an MSM network
# is_msm = ssppl.State(
#     'is_msm',
#     bool,
#     fill_value=ssu.sample('choice', par1=[True,False], par2=[0.1,0.9]),
#     eligibility='male'
# )
# msm_network = ss.msm(states=is_msm)
#
# # Make an MF network
# is_msf = ssppl.State(
#     'is_msf',
#     bool,
#     fill_value=ssu.sample('choice', par1=[True,False], par2=[0.1,0.9]),
#     eligibility='male'
# )
#
# #
# part_rates = {
#     'age': [15,20,25,30,35,40,45,50,55,60,65,70,75,80,85],
#     'm': {
#         'msm': 0.1,
#         'mf_marital': [0.1,0.5,0.6,0.7,0.8,0.9,0.8,0.7,0.7,0.6,0.5,0.5,0.5,0.5,0.5],
#         'mf_casual': 0.1,
#         ('msm', 'mf_marital'): 0.05,
#         ('msm', 'mf_casual'): 0.05,
#         ('mf_marital', 'mf_casual'): 0.05,
#         ('msm', 'mf_marital', 'mf_casual'): 0.05,
#     },
#     'f': {
#         'msm': 0,
#         'mf_marital': 0.7,
#         'mf_casual': 0.2,
#         ('msm', 'mf_marital'): 0,
#         ('msm', 'mf_casual'): 0,
#         ('mf_marital', 'mf_casual'): 0.1,
#         ('msm', 'mf_marital', 'mf_casual'): 0,
#     }
# }