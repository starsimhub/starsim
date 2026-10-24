"""Branching and p=."""
import starsim as ss, numpy as np, sciris as sc

def build(label, **kw):
    print(f'\n--- {label}')
    try:
        d = ss.Disease(name='d', **kw)
        d.describe()
        return d
    except Exception as e:
        print('  ERROR', type(e).__name__, str(e)[:300])

S, I, H, R, D = ss.states('S I H R D')
d1 = build('weighted destination, one dwell', infection=S >> I @ 0.3*I, course=I >> (0.1*H + 0.9*R) @ ss.days(7), discharge=H >> R @ ss.days(10), init_prev=0.01)
S, I, H, R, D = ss.states('S I H R D')
d2 = build('explicit list p=', infection=S >> I @ 0.3*I, course=ss.Transition(I, [H, R], p=[0.1, 0.9], dur=ss.days(7)), discharge=H >> R @ ss.days(10), init_prev=0.01)
d3 = build('arrow string with p=', infection=ss.transmission('S -> I', beta=0.3), course=ss.progression('I -> H | R', p=[0.1, 0.9], dur=7), discharge=ss.progression('H -> R', dur=10), init_prev=0.01)
S, I, H, R, D = ss.states('S I H R D')
d4 = build('per-branch dwell: two transitions with p=', infection=S >> I @ 0.3*I, severe=I >> 0.1*H @ ss.days(5), mild=I >> 0.9*R @ ss.days(10), discharge=H >> R @ ss.days(10), init_prev=0.01)
S, I, H, R, D = ss.states('S I H R D')
d5 = build('per-branch dwell plus a competing death hazard', infection=S >> I @ 0.3*I, severe=I >> 0.1*H @ ss.days(5), mild=I >> 0.9*R @ ss.days(10), death=I >> ss.null @ ss.perday(0.01), discharge=H >> R @ ss.days(10), init_prev=0.01)
S, I, H, R, D = ss.states('S I H R D')
d6 = build('p sums to 0.9 with no remainder transition', infection=S >> I @ 0.3*I, severe=I >> 0.1*H @ ss.days(5), mild=I >> 0.8*R @ ss.days(10), init_prev=0.01)
S, I, H, R, D = ss.states('S I H R D')
d7 = build('p sums to 0.1 with a timed transition taking the remainder', infection=S >> I @ 0.3*I, severe=I >> 0.1*H @ ss.days(5), mild=I >> R @ ss.days(10), init_prev=0.01)
S, I, H, R, D = ss.states('S I H R D')
d8 = build('a bare weight on a single dest with a rate (not dur): branch or hazard?', infection=S >> I @ 0.3*I, severe=I >> 0.1*H @ ss.perday(0.2), mild=I >> 0.9*R @ ss.perday(0.1), init_prev=0.01)
S, I, H, R, D = ss.states('S I H R D')
d9 = build('p= with no rate or dur: immediate', infection=S >> I @ 0.3*I, split=ss.Transition(I, [H, R], p=[0.1, 0.9]), discharge=H >> R @ ss.days(10), init_prev=0.01)
S, I, H, R, D = ss.states('S I H R D')
d10 = build('branch into null in one transition (E205 by design)', infection=S >> I @ 0.3*I, course=I >> (0.1*D + 0.9*R) @ ss.days(7), death=D >> ss.null @ ss.days(0.01), init_prev=0.01)
S, I, H, R, D = ss.states('S I H R D')
d11 = build('p as a function', infection=S >> I @ 0.3*I, course=ss.Transition(I, [H, R], p=[lambda sim: 0.1, None], dur=ss.days(7)), discharge=H >> R @ ss.days(10), init_prev=0.01)

# Are d1, d2, d3 equivalent? d4/d5 differ by design.
for name, d in dict(d1=d1, d2=d2, d3=d3, d4=d4, d5=d5).items():
    if d is None: continue
    sim = ss.Sim(sc.dcp(d), unit='day', stop=60, verbose=0).run()
    print(name, 'cum H entries:', sim.results.d.n_R[-1], 'peak H:', sim.results.d.n_H.max())

# ODE vs agents for the two-dwell branching model
print('\nd4 ode vs abm')
for kw in [dict(), dict(n_agents=2000)]:
    sim = ss.Sim(sc.dcp(d4), unit='day', stop=60, verbose=0, **kw).run()
    print(kw, sim.pars.solver, 'peak H', sim.results.d.n_H.max(), 'final R', sim.results.d.n_R[-1])
