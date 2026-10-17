import starsim as ss, sciris as sc, numpy as np
def mk(**kw): return ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir', **kw)
sc.heading('A: unitless sim time views')
s = ss.Sim(mk(), stop=10, verbose=0).run()
print('t:', s.t); print('timevec[:3]', s.timevec[:3], type(s.timevec)); print('res timevec', s.results.sir.n_I.timevec[:3]); print('unit', s.unit, 'unit_base', s.unit_base, 'dt', s.t.dt)
sc.heading('B: declared disease in ABM: state types')
s = ss.Sim(mk(), n_agents=200, stop=5, verbose=0).run()
d = s.diseases.sir
print('S type', type(d.S), d.S, '| sum', d.S.sum()); print('S.n in abm ->', getattr(d.S, 'n', 'AttributeError'))
print('state_list names', [x.name for x in d.state_list]); print('people.states', list(s.people.states.keys()))
print('people.sir?', hasattr(s.people, 'sir'))
sc.heading('C: results= kwarg')
try:
    s = ss.Sim(mk(), results=['n_I'], stop=5, verbose=0).run(); print(list(s.results.sir.keys()))
except Exception as E: print('ERROR', type(E).__name__, str(E)[:200])
sc.heading('D: run twice / E1003')
try:
    s = ss.Sim(mk(), stop=5, verbose=0).run(); s.run()
except Exception as E: print('ERROR', type(E).__name__, str(E)[:200])
sc.heading('E: sim.new on coded sim')
try:
    ss.Sim(ss.SIR(), networks='random', n_agents=100, stop=3, verbose=0).new()
except Exception as E: print('ERROR', type(E).__name__, str(E)[:200])
sc.heading('F: names from variables for other module types')
school = ss.RandomNet(); flu = ss.SIS()
s = ss.Sim([flu, school], n_agents=100, stop=3, verbose=0).run(); print(s.diseases.keys(), s.networks.keys())
sc.heading('G: ss.SIR as declared, same results?')
S, I, R, D = ss.states('S I R D')
beta = ss.par(ss.peryear(0.1))
class SIRd(ss.Disease):
    S, I, R, D = S, I, R, D
    infection = S >> I @ beta * I
    recovery = ss.progression('I -> R | D', dur=ss.lognorm_ex(mean=ss.years(6), std=1.0), p=[0.99, 0.01])
s = ss.Sim(SIRd(init_prev=0.01), networks=ss.RandomNet(), n_agents=1000, start=2000, stop=2020, verbose=0)
s.init(); print(s.solver, s.solver_reason); s.run(); print(list(s.results.sird.keys())); print('R0', s.diseases.sird.R0() if False else 'skip')
s2 = ss.Sim(SIRd(init_prev=0.01), start=2000, stop=2020, verbose=0).run(); print(s2.solver, s2.notes)
s2.describe()
