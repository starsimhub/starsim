"""A dwell (dur=) competing with a hazard, for people who enter the state during the run: as agents, the hazard gets about one step less exposure per entrant (monthly step: 0.076 vs 0.107)."""
import numpy as np, starsim as ss
def make(kind):
    S, A, B, C = ss.states('S A B C')
    dwell = dict(dur=ss.months(3)) if kind == 'dur' else dict(rate=ss.permonth(1/3)) # The same exponential, written two ways
    return ss.Disease(S, A, B, C, sa=S >> A @ ss.peryear(0.2), ab=ss.Transition(A, B, **dwell), ac=A >> C @ ss.permonth(0.04), name='d')
kw = dict(start=2000, stop=2010, dt=ss.months(1), unit='year', verbose=0, use_aging=False)
print('exact share to C:', round(0.04/(0.04 + 1/3), 3))
for kind in ['dur', 'rate']:
    for solver in ['abm', 'ode']:
        sim = ss.Sim(make(kind), solver=solver, **(dict(n_agents=50_000) if solver == 'abm' else dict(total_pop=50_000)), **kw).run()
        r = sim.results.d
        print(f'{kind:4s} {solver}: share to C {r.ac.sum()/(r.ac.sum() + r.ab.sum()):.3f}')

# The bias shrinks with the step, so it is about one step of the hazard per entrant
for label, dt in [('week', ss.weeks(1)), ('day', ss.days(1))]:
    sim = ss.Sim(make('dur'), solver='abm', n_agents=50_000, **{**kw, 'dt': dt, 'stop': 2003}).run()
    r = sim.results.d
    print(f'dur  abm, dt=1 {label}: share to C {r.ac.sum()/(r.ac.sum() + r.ab.sum()):.3f}')
