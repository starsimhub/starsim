"""A p= branch whose rate is gated by where= on another module's state never fires for people who entered while the condition was false."""
import starsim as ss
A, B = ss.states('A B')
clock = ss.Disease(A, B, tick=A >> B @ ss.constant(ss.months(3)), init=dict(A=1.0), name='clock') # Everyone reaches B at month 3
X, Y, Z = ss.states('X Y Z')
dec = ss.Disease(X, Y, Z,
    to_y = ss.Transition(X, Y, p=0.5, rate=ss.permonth(30), where=clock.B), # Split on entry, then move as soon as clock.B holds
    to_z = ss.Transition(X, Z, p=0.5, rate=ss.permonth(30), where=clock.B),
    init=dict(X=1.0), name='dec')
sim = ss.Sim([clock, dec], n_agents=1000, start=2000, stop=2001, dt=ss.months(1), unit='year', verbose=0).run()
r = sim.results
print('n_B   ', r.clock.n_B[:8].astype(int))
print('n_X   ', r.dec.n_X[:8].astype(int), '(expected 0 from month 4)')
# The same with plain hazards (no p=) works:
X, Y = ss.states('X Y')
A, B = ss.states('A B')
clock = ss.Disease(A, B, tick=A >> B @ ss.constant(ss.months(3)), init=dict(A=1.0), name='clock')
dec = ss.Disease(X, Y, to_y=ss.Transition(X, Y, rate=ss.permonth(30), where=clock.B), init=dict(X=1.0), name='dec')
sim = ss.Sim([clock, dec], n_agents=1000, start=2000, stop=2001, dt=ss.months(1), unit='year', verbose=0).run()
print('hazard', sim.results.dec.n_X[:8].astype(int))
