""" Follow-ups: masked AttributeError in modify.active; .base; transfer vs vaccinate; verbosity """
import traceback
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(verbose=0, unit='day', total_pop=20_000, stop=120)

def make(**kwargs):
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> S', dur=10), init_prev=0.01, name='sis', **kwargs)

print('=== 1. the masked AttributeError')
sis = make()
effect = ss.modify(sis.pars.beta, multiply=0.2, when=lambda sim: sim.interventions.closure.closed)
try:
    ss.Sim(sis, interventions=effect, solver='ode', **kw).run()
except Exception as E:
    print(type(E).__name__, ':', E)
    tb = traceback.format_exc().splitlines()
    print('\n'.join(tb[-8:]))

print('\n=== 2. what is arr.base?')
sis = make()
sim = ss.Sim(sis, solver='abm', n_agents=200, **kw).init()
arr = sim.diseases.sis.rel_sus
print(type(arr.base), repr(arr.base)[:80])

print('\n=== 3. transfer+adjust vs vaccinate: the same model?')
def A(solver):
    sis = make().stratify(vaccinated=[False, True], adjust={'infection': [1.0, 0.1]})
    return ss.Sim(sis, interventions=ss.transfer(vaccinated=True, coverage=0.8, years=[10]), solver=solver, n_agents=2000, **kw).run()
def B(solver):
    sis = make()
    return ss.Sim(sis, interventions=ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.8, years=[10]), solver=solver, n_agents=2000, **kw).run()
def C(solver):
    sis = make().stratify(vaccinated=[False, True])
    mod = ss.modify(sis.infection, multiply=0.1, where=sis.vaccinated == True)
    return ss.Sim(sis, interventions=[ss.transfer(vaccinated=True, coverage=0.8, years=[10]), mod], solver=solver, n_agents=2000, **kw).run()
def D(solver):
    sis = make().stratify(vaccinated=[False, True])
    mod = ss.modify(sis.rel_sus, multiply=0.1, where=sis.vaccinated == True)
    return ss.Sim(sis, interventions=[ss.transfer(vaccinated=True, coverage=0.8, years=[10]), mod], solver=solver, n_agents=2000, **kw).run()
for solver in ['ode', 'abm']:
    a, b, c, d = A(solver), B(solver), C(solver), D(solver)
    f = lambda s: f'n_I[60]={s.results.sis.n_I[60]:7.0f} n_I[11]={s.results.sis.n_I[11]:7.0f} delivered={s.results[[m.name for m in s.interventions()][0]].delivered.sum():7.0f}'
    print(solver, '\n  A transfer+adjust(infection):', f(a), '\n  B vaccinate(efficacy=0.9):  ', f(b), '\n  C transfer+modify(infection):', f(c), '\n  D transfer+modify(rel_sus): ', f(d))
sim = A('ode')
print('A describe:'); sim.describe()
