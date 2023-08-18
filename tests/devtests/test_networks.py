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


part_rates = {
    'm': {
        'msm': 0.1,
        'mf_marital': 0.6,
        'mf_casual': 0.1,
        ('msm', 'mf_marital'): 0.05,
        ('msm', 'mf_casual'): 0.05,
        ('mf_marital', 'mf_casual'): 0.05,
        ('msm', 'mf_marital', 'mf_casual'): 0.05,
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


