""" Run the code written by to_code() for the overview model """
import runpy, sys
ns = runpy.run_path('/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/spec/overview_code.py', run_name='notmain')
sim = ns['sim']
sim.pars.verbose = 0
sim.init()
print('generated code builds a sim; start/stop:', sim.t.start, sim.t.stop)
import starsim as ss
print(sim.hash())
