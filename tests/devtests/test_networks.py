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

# Create an MSM state
msm_distdict = dict(dist='choice', par1=[True, False], par2=[0.15, 0.85])
msm = ss.State('msm', bool, distdict=msm_distdict, eligibility='male')

# Create people with MSM
ppl = ss.People(100, states=msm)

# Give people non-zero ages
ppl.age = norm.rvs(30, 10, 100)

# Add an FSW state
def fsw_eligible(people): return (people.age > 20) & (people.age < 30) & people.female
fsw_distdict = dict(dist='choice', par1=[True, False], par2=[0.15, 0.85])
ppl.add_state('fsw', bool, eligibility=fsw_eligible, distdict=fsw_distdict, na_val=False)


# Q: should networks add states in a similar way to modules?
# e.g.
#   class msm(ss.Network):
#       def __init__():
#           self.states = ssu.named_dict(State('msm', bool, True))
#   people.add_network(msm())

class msm(ss.Network):
    pass
