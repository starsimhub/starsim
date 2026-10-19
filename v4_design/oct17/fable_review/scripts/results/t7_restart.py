"""restart_runs path vs copies: what the MultiSim holds"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

def make(beta=0.3):
    return ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', dur=10), init=dict(I=3), name='sir')
kw = dict(total_pop=300, unit='day', stop=40, verbose=0, solver='gillespie')

# A: restart path
A = ss.Sim(make(), **kw).run(n_runs=5)
print('A: len(msim)', len(A), 'n sims', len(A.sims), 'runs in results', A.results.sir.n_I.values.shape[0])
print('A: runs identical?', [np.allclose(A.results.sir.n_I.values[0], A.results.sir.n_I.values[i]) for i in range(1,5)])
print('A: summarize n_I', A.summarize()['sir_n_I'])
print('A: compare', A.compare(output=True).shape)
A.reduce()
r = A.results.sir_n_i
print('A: reduced low==high everywhere?', np.allclose(r.low, r.high), ' == values?', np.allclose(r.low, r.values))
# B: the copies path (forced by passing an option that is not serial, e.g. shrink=False)
B = ss.parallel(ss.Sim(make(), **kw), n_runs=5, shrink=False)
print('B: len(msim)', len(B), 'results type', type(B.results).__name__, 'runs', B.results.sir.n_I.values.shape)
print('B: summarize n_I', B.summarize()['sir_n_I'])
fig = B.plot(); print('B: unreduced plot lines axis0', len(fig.axes[0].lines))
B.reduce()
try:
    fig = B.plot(); print('B: reduced plot ok')
except Exception as e:
    print('B: reduced plot ERR', type(e).__name__, str(e)[:100])
print('B: results keys after reduce', list(B.results.keys()))
# C: Starsim 3 sim reduce + plot
C = ss.MultiSim(ss.Sim(diseases='sis', networks='random', n_agents=300, start=2000, stop=2002, verbose=0), n_runs=3).run()
C.reduce()
try:
    fig = C.plot(); print('C: ss3 reduced plot ok, results type', type(C.results).__name__)
except Exception as e:
    print('C: reduced plot ERR', type(e).__name__, str(e)[:100])
# D: data outside the period
sim = ss.Sim(make(), total_pop=300, start='2020-01-01', stop='2020-02-01', verbose=0,
             data=pd.DataFrame(dict(date=['2023-01-08', '2023-01-15'], infection=[5, 9])))
try:
    sim.run(); print('D: ran fine with data outside the period')
    ss.calibration.compare(sim)
except Exception as e:
    print('D:', type(e).__name__, str(e)[:200])
# E: infection log with a network
sim = ss.Sim(make(), networks='random', n_agents=300, unit='day', stop=20, analyzers='infection_log', verbose=0).run()
print('E: log rows with network', len(sim.analyzers.infection_log.to_df()), 'infections', sim.results.sir.infection.sum())
sim = ss.Sim(make(), n_agents=300, unit='day', stop=20, analyzers='infection_log', verbose=0).run()
print('E: log rows no network', len(sim.analyzers.infection_log.to_df()), 'infections', sim.results.sir.infection.sum(), 'solver', sim.solver)
# F: a Starsim-3 disease alongside a declared one: result names
sim = ss.Sim([make(), ss.SIS(beta=0.1)], networks='random', n_agents=300, unit='day', stop=10, verbose=0).run()
print('F: sir keys', [k for k in sim.results.sir.keys() if k!='timevec'])
print('F: sis keys', [k for k in sim.results.sis.keys() if k!='timevec'])
# G: reading a result during the run (so_far)
class Watch(ss.Analyzer):
    def step(self):
        r = self.sim.results.sir.infection
        if self.ti == 5:
            print('G: at ti=5, infection[:ti]', r[:self.ti], ' full len', len(r), 'last 3 so far', r[max(0,self.ti-3):self.ti])
sim = ss.Sim(make(), analyzers=Watch(), solver='binomial', **kw).run()
