"""SIR four ways: do they build the same model?"""
import sciris as sc
import starsim as ss
import numpy as np

# 1. arrow strings
sir1 = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.3),
    recovery  = ss.progression('I -> R', dur=10),
    init_prev = 0.01, name='sir',
)

# 2. operators
beta = 0.3
S, I, R = ss.states(3)
infection = S >> I @ beta * I
recovery  = I >> R @ ss.days(10)
sir2 = ss.Disease(infection, recovery, init_prev=0.01, name='sir')

# 3. explicit
S3, I3, R3 = ss.states('S I R')
sir3 = ss.Disease(
    ss.Transition(S3, I3, rate=0.3 * I3, name='infection'),
    ss.Transition(I3, R3, dur=10, name='recovery'),
    init_prev=0.01, name='sir',
)

# 4. class
class SIR(ss.Disease):
    S, I, R   = ss.states(3)
    infection = S >> I @ 0.3 * I
    recovery  = I >> R @ ss.days(10)
sir4 = SIR(init_prev=0.01, name='sir')

sims = {}
for k, d in dict(strings=sir1, operators=sir2, explicit=sir3, cls=sir4).items():
    print(f'\n===== {k}')
    d.describe()
    print(d.transitions)
    print('pars:', dict(d.pars))
    j = d.to_json()
    print('json transitions:', [(t['name'], t['rate'], t['dur']) for t in j['transitions']])
    print('json pars:', [(p['name'], p['value']) for p in j['pars']])
    sim = ss.Sim(d, stop=100, unit='day', verbose=0).run()
    sims[k] = sim
    print('final R:', sim.results.sir.n_R[-1], 'solver:', sim.solver if hasattr(sim,'solver') else sim.pars.solver)

print('\nresults identical across spellings?')
ref = sims['strings'].results.sir.n_R[:]
for k, s in sims.items():
    print(k, np.allclose(ref, s.results.sir.n_R[:]))

# Unit issue: sir1 has dur=10 bare, sir2 has ss.days(10): with unit='day' they agree. Without unit=?
print('\n--- without unit= ---')
for k, d in dict(strings=sir1, operators=sir2).items():
    d = sc.dcp(d)
    sim = ss.Sim(d, stop=100, verbose=0)
    sim.init()
    print(k, 'unit:', sim.unit, 'dt:', sim.t.dt)
