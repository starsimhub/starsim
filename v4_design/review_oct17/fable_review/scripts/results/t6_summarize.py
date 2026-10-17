"""sim.summarize for declared flows; results=; infection_log"""
import numpy as np
import sciris as sc
import starsim as ss

def make():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.02, name='sir')
kw = dict(total_pop=1000, unit='day', stop=30, verbose=0)
sim = ss.Sim(make(), **kw).run()
s = sim.summarize()
print({k: round(v, 2) for k, v in s.items()})
r = sim.results.sir
print('infection sum', r.infection.sum(), 'mean', r.infection.mean(), ' cum_infection[-1]', r.cum_infection[-1])
print('sim.to_df cols', list(sim.to_df().columns))
# results=
try:
    sim = ss.Sim(make(), results=['n_I'], **kw).run()
    print('results=[n_I] keys', list(sim.results.sir.keys()))
except Exception as e:
    print('results= ERR', type(e).__name__, str(e)[:300])
# infection_log
try:
    sim = ss.Sim(make(), analyzers='infection_log', **kw).run()
    print('infection_log solver', sim.solver, 'rows', len(sim.analyzers.infection_log.to_df()))
except Exception as e:
    print('infection_log ERR', type(e).__name__, str(e)[:300])
try:
    sim = ss.Sim(make(), analyzers='infection_log', solver='ode', **kw).run()
    print('ode + infection_log: ran, rows', len(sim.analyzers.infection_log.to_df()))
except Exception as e:
    print('ode+infection_log ERR', type(e).__name__, str(e)[:300])
# so_far: reading a result during the run
# resample of a flow vs stock
sim = ss.Sim(make(), start='2020-01-01', stop='2020-03-01', total_pop=1000, verbose=0).run()
r = sim.results.sir
print('infection.resample(week)[:3]', r.infection.resample('week').values[:3], 'n_I.resample(week)[:3]', r.n_I.resample('week').values[:3], 'prevalence.resample', r.prevalence.resample('week').values[:3])
print('n_I weekly = mean, not end-of-week?  n_I[7]=', r.n_I[7], 'mean of n_I[1:8]=', r.n_I[1:8].mean(), 'mean n_I[0:7]', r.n_I[0:7].mean())
print('cum_infection.resample(week)', r.cum_infection.resample('week').values[:3])
# Result repr
print(repr(r.infection)[:200])
print(str(r.infection))
print(r)
