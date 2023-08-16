"""
Experimenting with networks
Key properties of networks:
    - who participates? consider eligibility vs uptake, e.g. all females 15-49 *can* be FSW but not all are
    - how do participants in the network choose their partners?
    - are participants in one network restricted from participating in other networks?

"""

import stisim as ss
from scipy.stats import norm

####################################
# Adding states to people
####################################

msm = ss.msm()
ppl = ss.People(100)
ppl.age = norm.rvs(30, 10, 100)  # Give people non-zero ages
ppl.add_network(msm)

# Sexually active MSM
ppl.networks['msm'].active(ppl)
# What about ppl.msm.active? Should ppl.msm be an objdict of states (as it is now), or the network itself?


