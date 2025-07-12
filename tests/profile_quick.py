"""
Quickly profile a sim run
"""
import starsim as ss

sim = ss.Sim(networks='random', diseases='sis')

prof = sim.profile(follow=[ss.Sim.init, ss.Sim.run])
prof.disp()