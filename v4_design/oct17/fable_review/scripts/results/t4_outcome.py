"""ss.outcome: delay vs duration, every"""
import numpy as np
import sciris as sc
import starsim as ss

def make(**oc):
    S, I, R = ss.states(3)
    infection = S >> I @ 0.3 * I
    recovery  = I >> R @ ss.days(10)
    hosp = ss.outcome(0.1 * infection, **oc)
    return ss.Disease(infection, recovery, hosp, init_prev=0.02, name='sir')

kw = dict(total_pop=1000, unit='day', stop=60, verbose=0)
# delay exact, duration exponential?
for oc in [dict(delay=ss.days(7)), dict(duration=ss.days(7)), dict(duration=ss.constant(7, unit='days')), dict(delay=ss.days(7), duration=ss.days(7))]:
    sim = ss.Sim(make(**oc), **kw).run()
    r = sim.results.sir
    inf = r.infection.values
    hosp = r.hosp.values
    # lag of peak
    print(f'{str(oc):70s} peak inf t={inf.argmax()}, peak hosp t={hosp.argmax()}, ', end='')
    if 'n_hosp' in r:
        n = r.n_hosp.values
        # mean stay = sum(n)/sum(hosp) roughly
        print(f'n_hosp peak t={n.argmax()}, implied mean stay={n.sum()/hosp.sum():.2f}, final n_hosp={n[-1]:.2f}', end='')
    print(' keys', [k for k in r.keys() if 'hosp' in k])

# every
sim = ss.Sim(make(every=ss.weeks(1)), **kw).run()
h = sim.results.sir.hosp.values
print('every=week: nonzero at steps', np.nonzero(h)[0][:6], 'sum hosp/sum inf', h.sum()/sim.results.sir.infection.sum())
print('cum_hosp exists?', 'cum_hosp' in sim.results.sir)
print('resample(week) of weekly outcome:', sim.results.sir.hosp.resample('week').values[:4])
# plot of an 'every' outcome
fig = sim.results.sir.hosp.plot()
print('hosp.plot title', fig.axes[0].get_title())

# abm
sim = ss.Sim(make(delay=ss.days(7), duration=ss.days(7)), solver='abm', n_agents=1000, **kw).run()
r = sim.results.sir
print('abm: keys', [k for k in r.keys() if 'hosp' in k], 'hosp sum', r.hosp.sum(), 'n_hosp max', r.n_hosp.max())

# outcome on the sim
flu = make(delay=ss.days(1))
try:
    sim = ss.Sim(flu, ili=ss.outcome(0.3 * flu.transitions['infection'] if hasattr(flu, 'transitions') else 'infection'), **kw).run()
    print('sim-level outcome keys', [k for k,v in sim.results.items() if isinstance(v, ss.Result)])
except Exception as e:
    print('sim-level outcome ERR', type(e).__name__, str(e)[:300])
print('Disease attrs with transitions:', [a for a in dir(flu) if 'trans' in a.lower()][:10])
