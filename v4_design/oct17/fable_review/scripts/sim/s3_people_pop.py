import starsim as ss, sciris as sc, numpy as np
def mk(**kw): return ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, **kw)
def try_(label, fn):
    sc.heading(label)
    try:
        return fn()
    except Exception as E:
        print('ERROR', type(E).__name__, str(E)[:400])

sc.heading('A: people in an ODE run')
s = ss.Sim(mk(), stop=10, verbose=0).run()
p = s.people
print('type', type(p), 'len', len(p), 'n_agents', p.n_agents, 'states', list(p.states.keys()))
print('pars.n_agents', s.pars.n_agents, 'total_pop', s.pars.total_pop, 'pop_scale', s.pars.pop_scale)
try_('  ppl.age in ODE', lambda: print(p.age))
try_('  ppl.female in ODE', lambda: print(p.female))
try_('  ppl.alive in ODE', lambda: print(p.alive, p.alive.sum()))
print('sir.S in ODE:', type(s.diseases.disease.S), s.diseases.disease.S)
try_('  sir.S.n', lambda: print(s.diseases.disease.S.n))

sc.heading('B: condition on age in ODE: needs ss.People(1)?')
sir = mk(name='sir')
try_('  lazy cond with no People', lambda: ss.Sim(sir, stratify=dict(age=[0,15,65]), connectors=ss.modify(sir.pars.beta, multiply=0.5, where=lambda sim: sim.people.age < 15), stop=10, verbose=0).run().describe())
sir = mk(name='sir')
ppl = ss.People(1)
try_('  lazy cond with ss.People(1)', lambda: ss.Sim(sir, stratify=dict(age=[0,15,65]), connectors=ss.modify(sir.pars.beta, multiply=0.5, where=ppl.age < 15), stop=10, verbose=0).run().describe())
sir = mk(name='sir')
try_('  lazy cond with ss.People(1) but age cut at 10 (not a bin edge)', lambda: ss.Sim(sir, stratify=dict(age=[0,15,65]), connectors=ss.modify(sir.pars.beta, multiply=0.5, where=ppl.age < 10), stop=10, verbose=0).run())

sc.heading('C: total_pop / n_agents defaults')
for kw in [dict(), dict(total_pop=50e6), dict(n_agents=500), dict(n_agents=500, total_pop=50e6), dict(total_pop=50e6, solver='abm'), dict(total_pop=500, solver='abm'), dict(people=ss.People(300)), dict(people=ss.People(300), total_pop=50e6), dict(pop_scale=10), dict(pop_scale=10, solver='abm')]:
    try:
        s = ss.Sim(mk(), stop=5, verbose=0, **kw).run()
        print(f'{str(kw):55s} -> solver={s.solver:4s} n_agents={s.pars.n_agents:>8} total_pop={s.pars.total_pop:>12,.0f} pop_scale={s.pars.pop_scale:g} n_S0={s.results.disease.n_S[0]:,.0f} len(people)={len(s.people)}')
    except Exception as E:
        print(f'{str(kw):55s} -> ERROR {type(E).__name__}: {str(E)[:150]}')
sc.heading('C2: coded SIR with total_pop')
s = ss.Sim(ss.SIR(), networks=ss.RandomNet(), total_pop=50e6, stop=5, verbose=0).run()
print('coded total_pop only ->', s.pars.n_agents, s.pars.total_pop, s.pars.pop_scale)
