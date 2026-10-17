""" Threshold-triggered interventions: when= and a class with memory, as the spec writes them """
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(verbose=0, unit='day', total_pop=20_000, stop=200)

def make():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.001, name='sir')

print('=== 1. bare when= threshold: flickers?')
sir = make()
calls = []
def many(sim):
    out = sim.results.sir.n_I[sim.ti - 1] > 1000
    calls.append(bool(out))
    return out
sim = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.3, when=many), solver='ode', **kw).run()
flips = np.sum(np.diff(np.array(calls, dtype=int)) != 0)
print('on/off flips:', flips, ' steps on:', sum(calls), '/', len(calls))

print('\n=== 2. the spec\'s SchoolClosure class verbatim (08-interventions.md), does it run?')
class SchoolClosure(ss.Intervention):
    def __init__(self, threshold=1000, hold=ss.weeks(4), **kwargs):
        super().__init__(**kwargs)
        self.define_pars(threshold=threshold, hold=hold)
        self.closed      = False
        self.date_closed = None

    def step(self):
        sim = self.sim
        new = sim.results.sir.infection   # spec says new_infections; the result is named after the transition
        many = new[max(0, sim.ti - 7):sim.ti].sum() > self.pars.threshold
        if many and not self.closed:
            self.closed, self.date_closed = True, sim.now
        elif self.closed and not many and sim.now - self.date_closed >= self.pars.hold:
            self.closed = False

sir = make()
closure = SchoolClosure()
effect  = ss.modify(sir.pars.beta, multiply=0.2, when=lambda sim: sim.interventions.closure.closed)
try:
    sim = ss.Sim(sir, interventions=[closure, effect], **kw).run()
    print('ran; solver =', sim.solver, '; closure name =', closure.name, '; closed at end:', closure.closed)
except Exception as E:
    print('FAILED:', type(E).__name__, str(E)[:500])

print('\n=== 2b. with compartmental = True and sim.ti arithmetic')
class Closure2(SchoolClosure):
    compartmental = True
sir = make()
closure = Closure2(name='closure')
effect  = ss.modify(sir.pars.beta, multiply=0.2, when=lambda sim: sim.interventions.closure.closed)
try:
    sim = ss.Sim(sir, interventions=[closure, effect], **kw).run()
    print('ran; solver =', sim.solver, '; closed at end:', closure.closed, '; peak n_I', sim.results.sir.n_I.max())
except Exception as E:
    print('FAILED:', type(E).__name__, str(E)[:500])

print('\n=== 3. does the spec\'s `sim.results.sir.new_infections` exist?')
sir = make()
sim = ss.Sim(sir, **kw).init()
print([k for k in sim.results.sir.keys()])
print('sim.now:', repr(sim.now), type(sim.now))
