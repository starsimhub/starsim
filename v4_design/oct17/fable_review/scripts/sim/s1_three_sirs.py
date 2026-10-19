import starsim as ss, sciris as sc, numpy as np
sc.heading('A: ss.Sim(ss.SIR())')
try:
    s = ss.Sim(ss.SIR(), n_agents=1000, stop=30, verbose=0).run()
    print(repr(s)); print(list(s.results.sir.keys())); print(s.networks.keys())
except Exception as E:
    print('ERROR', type(E).__name__, E)

sc.heading('B: ss.Sim(ss.SIR(), networks=ss.RandomNet())')
s = ss.Sim(ss.SIR(), networks=ss.RandomNet(), n_agents=1000, stop=30, verbose=0).run()
print(repr(s)); print(list(s.results.sir.keys())); print('time', s.t.start, s.t.stop, s.t.dt, s.t.unit if hasattr(s.t,'unit') else None, len(s.t))
print('pars:', dict(s.diseases.sir.pars))
s.describe()

sc.heading('C: declared SIR')
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
s2 = ss.Sim(sir, stop=30, verbose=0).run()
print(repr(s2)); print(list(s2.results.sir.keys())); print('time', s2.t.start, s2.t.stop, s2.t.dt, len(s2.t))
s2.describe()

sc.heading('D: declared SIR as agents')
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
s3 = ss.Sim(sir, n_agents=1000, stop=30, verbose=0).run()
print(repr(s3)); print(list(s3.results.sir.keys()))
print('people results', [k for k in s3.results.keys()])
print('people results (ode)', [k for k in s2.results.keys()])
print('people results (abm coded)', [k for k in s.results.keys()])
s3.describe()

sc.heading('E: demo')
s4 = ss.demo(plot=False, verbose=0)
print(repr(s4)); print(list(s4.results.sir.keys()))
