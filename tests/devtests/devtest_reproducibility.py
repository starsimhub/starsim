import starsim as ss
import sciris as sc
import numpy as np

ss.options(multirng=True)

class use_np_rands(ss.Intervention):
    def apply(self, sim):
        print(np.random.rand())
        
class use_ss_rands(ss.Intervention):
    def apply(self, sim):
        print(ss.uniform.rvs(size=1))

pars = dict(diseases=dict(type='sir', beta=dict(random=[0.05, 0.05])), networks='random', n_agents=10e3)

sims = [
    ss.Sim(pars=pars, label='a'),
    ss.Sim(pars=pars, label='b', interventions=use_np_rands()),
    ss.Sim(pars=pars, label='c', interventions=use_ss_rands()),
]

for sim in sims:
    sim.run()

for i,sim in enumerate(sims):
    sc.heading(f'Sim {i}')
    print(sim.summarize())