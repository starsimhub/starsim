"""
Quickly profile a sim run
"""
import starsim as ss

net = ss.RandomNet()
sis = ss.SIS()
sim = ss.Sim(networks=net, diseases=sis)
prof = sim.profile(follow=[net.add_pairs, sis.infect])