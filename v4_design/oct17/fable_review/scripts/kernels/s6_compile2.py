"""After the SIR has compiled (warm cache from s3), do other model shapes trigger further compiles? And is anything printed at default verbosity during a compile?"""
import time, sys
import numpy as np
import starsim as ss
sys.path.insert(0, '/home/cliffk/idm/ssv4/tests')
from test_kernels import models
import sciris as sc

for key in ['sir', 'seir', 'ages', 'towns', 'malaria', 'tb']:
    make, pars = models[key]
    sim = ss.Sim(make(), **sc.mergedicts(pars, dict(verbose=0))).init()
    t = time.perf_counter()
    sim.run()
    el = time.perf_counter() - t
    print(f'{key:8s} first run {el:7.2f} s  kernel_only={sim.kernel_only()}')
    t = time.perf_counter()
    ss.Sim(make(), **sc.mergedicts(pars, dict(verbose=0))).run()
    print(f'{key:8s} second    {time.perf_counter() - t:7.3f} s')
for solver in ['gillespie', 'tau', 'sde']:
    make, pars = models['sir']
    t = time.perf_counter()
    ss.Sim(make(), **sc.mergedicts(pars, dict(verbose=0, solver=solver))).run()
    print(f'{solver:9s} first run {time.perf_counter() - t:7.2f} s')
