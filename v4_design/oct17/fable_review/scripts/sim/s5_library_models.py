import starsim as ss, sciris as sc, numpy as np, warnings
sc.heading('A: ss.SIR() as a library model: describe/equations/solvers/R0')
sir = ss.SIR()
print(repr(sir)); print('graph:', sir.graph)
for m in ['describe', 'equations', 'diagram', 'R0']:
    try:
        out = getattr(sir, m)(); print(m, '->', out)
    except Exception as E:
        print(m, '-> ERROR', type(E).__name__, str(E)[:200])
s = ss.Sim(ss.SIR(), networks=ss.RandomNet(), n_agents=500, stop=5, verbose=0)
s.solvers()
try:
    ss.Sim(ss.SIR(), networks=ss.RandomNet(), n_agents=500, stop=5, verbose=0, solver='ode').run()
except Exception as E:
    print('solver=ode on ss.SIR -> ERROR', type(E).__name__, str(E)[:300])
sc.heading('B: ss.Sim(ss.SIR()) with no network: silent?')
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    s = ss.Sim(ss.SIR(), n_agents=500, stop=20, verbose=0).run()
    print('warnings:', [str(x.message)[:100] for x in w])
print('n_infected:', s.results.sir.n_infected[[0,5,10,20]], 'new_infections total', s.results.sir.new_infections.sum())
sc.heading('C: sim.to_spec() of a coded model, and hash')
print(sc.pp(s.to_spec()['modules'][0]))
print(s.hash())
sc.heading('D: Starsim 3 pars dict as first positional + model positional both?')
try:
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
    s = ss.Sim(dict(stop=10, verbose=0), diseases=sir).run(); print(repr(s))
except Exception as E:
    print('ERROR', type(E).__name__, str(E)[:300])
sc.heading('E: ss.Sim repr/people for ODE and n=1')
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
s = ss.Sim(sir, stop=10, verbose=0).run()
print(repr(s)); print(s.people); print(len(s)); print(s.summarize())
