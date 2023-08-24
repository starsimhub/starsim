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
import numpy as np

####################################
# Adding states to people
####################################

msm = ss.msm()
ppl = ss.People(100)
ppl.age = norm.rvs(30, 10, 100)  # Give people non-zero ages
ppl.add_network(msm)

####################################
# Experiments with representing MSM
####################################
people = ss.People(100)
# Old way:
# Networks were stored in a dictionary of networks
# people.contacts['msm']

# Possible altenrative
# people.routes.sexual is a single network that aggregates all subnetworks
# We may also have
# people.routes.injecting  # second type of network for STIsim
# people.routes.airborne  # if modeling TB for e.g.

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


# Examples
unmarried = lambda ppl: ppl.networks.sexual_contacts.mf_marital.current_partners == 0
networks = [
    msm(pars=dict(part_rate=0.1)),
    mf_casual(pars=dict(part_rate=0.3), eligibility=unmarried),
    # Participation rates for these layers might be age/sex dependent
    mf_marital(pars=dict(part_rate=0.7))
]
st = SexualTransmission(networks=networks)


# This still hasn't solved the issue of how to define cross-network mixing

#########################################################
# Alternative method using only 1 sexual network and defining sexual orientation
#########################################################
class orientation:
    def __init__(self, options=None, pop_shares=None, m_partner_prob=None):
        self.options = options
        self.pop_shares = pop_shares
        self.m_partner_prob = m_partner_prob


# Motivating examples:
# 1. Sexual orientation patterns from Western countries
western_prefs = orientation(
    options=[0, 1, 2, 3, 4],  # Abridged Kinsey scale
    m_partner_prob=[0.00, 0.005, 0.020, 0.100, 1.00],  # Probability of taking a male partner
    pop_shares=[0.93, 0.040, 0.005, 0.005, 0.02]
    # From https://en.wikipedia.org/wiki/Demographics_of_sexual_orientation
)
orientation1 = ssppl.State('orientation', int,
                           distdict=dict(dist='choice', par1=western_prefs.m_partner_prob,
                                         par2=western_prefs.pop_shares),
                           eligibility='male')

# 2. Sexual orientation with no MSM/MF mixing
exclusive_prefs = orientation(
    options=['mf', 'msm'],  # No mixing
    m_partner_prob=[0.0, 1.0],  # Probability of taking a male partner
    pop_shares=[0.95, 0.05]
)

# 3. Sexual orientation without open MSM
closeted_prefs = orientation(
    options=[0, 1, 2, 3],  # last category still only partners with MSM 50% of the time
    m_partner_prob=[0.0, 0.005, 0.02, 0.50],  # Probability of taking a male partner
    pop_shares=[0.9, 0.010, 0.03, 0.06]
)

# Now take the orientation distribution and pass it into the network.
# Use a SexualNetwork class which has attributes like orientation (which other networks wouldn't have)


class SexualNetwork(ss.Network):
    def __init__(self, pars=None, states=None):
        self.pars = ssu.omerge(dict(
            part_rate=1,
            orientation=closeted_prefs,
        ), pars)
        self.states = ssu.omerge(self.states, states)

        # Add orientation
        opars = self.pars['orientation']
        odist = dict(dist='choice', par1=opars.m_partner_prob, par2=opars.pop_shares)
        self.states['orientation'] = ssppl.State('orientation', float, distdict=odist)

    def initialize(self, p):
        """ Do other initializations that need people - this also adds self.oreintation as an initialized state """
        return

    def add_pairs(self, p):
        """
        Partner-finding needs to account for sexual orientation.
        Q, how can we incorporate age and other preferences into this?
        """
        male_uids = people.uid[people.male]

        # Find males who have fewer than their desired number of partners
        # Figure out who they want to partner with using self.orientation
        # Questions:
        #   - should there be assortativity between orientation, i.e. 6 more likely to pair with 6?
        #   - how can we incorporate differences in duration/concurrency with age/orientation?


#########################################################
# Older ideas
#########################################################
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

# here, sim.people.msm is NOT a network
# to find MSM, use
# sim.people.networks.sexual_contacts.msm.participant
# sim.people.is_msm # refers to sim.people.networks.sexual_contacts.participant
#
# Ideas for other kinds of networks
# A school network
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
