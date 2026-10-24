""" Screen-then-treat cascade: naming, results, cases = sum of diagnosed, and the old API alongside """
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(verbose=0, unit='day', total_pop=20_000, stop=100)

def make():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> S', dur=10), init_prev=0.01, name='sis')

print('=== 1. screen -> confirm -> treat, with a delay')
for solver in ['ode', 'abm']:
    sis = make()
    rapid = ss.Test(sensitivity=0.9, specificity=0.95, cost=1.0)
    pcr = ss.Test(sensitivity=0.99, specificity=0.99, delay=ss.days(2), cost=10.0)
    drug = ss.Treatment(efficacy=0.95, cost=20.0)
    confirm = ss.screen(pcr, coverage=0.9, on_positive=ss.treat(drug, coverage=0.8))
    screening = ss.screen(rapid, coverage=0.4, interval=ss.days(7), start=20, on_positive=confirm)
    sim = ss.Sim(sis, interventions=screening, solver=solver, n_agents=2000, **kw).run()
    names = [m.name for m in sim.interventions()]
    print(solver, 'interventions:', names)
    print('   strats:', [(s.name, len(s.labels)) for s in sim.strats])
    print('   confirm.name is now:', confirm.name, '| results keys:', list(sim.results.keys()))
    print('   diagnosed by screening:', sim.results.screening.diagnosed.sum(), ' by confirm:', sim.results[names[1]].diagnosed.sum(), ' treated:', sim.results[names[-1]].delivered.sum())
    print('   total cost:', sum(sim.results[n].cost.sum() for n in names if 'cost' in sim.results[n]))
    print('   product result keys:', list(sim.results.rapid.keys()))

print('\n=== 2. cases = a.diagnosed + b.diagnosed, as the spec writes it (before the run)')
sis = make()
a = ss.screen(ss.Test(sensitivity=0.8), on=sis.infection, coverage=0.5, name='symptomatic')
b = ss.screen(ss.Test(sensitivity=0.9), coverage=0.1, interval=7, name='screening')
try:
    cases = a.diagnosed + b.diagnosed
    print('declared cases:', cases)
except Exception as E:
    print('FAILS before the run:', type(E).__name__, str(E)[:200])
sim = ss.Sim(sis, interventions=[a, b], solver='ode', **kw).run()
try:
    cases = sim.results.symptomatic.diagnosed + sim.results.screening.diagnosed
    print('after the run: cases total', cases.sum(), type(cases).__name__)
except Exception as E:
    print('FAILS after the run:', type(E).__name__, str(E)[:200])

print('\n=== 3. default eligibility of a treatment: infected people; of a screen: everyone')
sis = make()
treatment = ss.treat(ss.Treatment(efficacy=1.0), coverage=1.0, years=[20])
sim = ss.Sim(sis, interventions=treatment, solver='ode', **kw).run()
print('treated at 20:', sim.results.treatment.delivered[20], ' n_I[19]:', sim.results.sis.n_I[19], ' n_I[20]:', sim.results.sis.n_I[20])

print('\n=== 4. Starsim 3 API still importable and runs alongside (routine_vx + simple_vx) on an ss.SIR')
sir3 = ss.SIR(beta=0.05, init_prev=0.01)
vx = ss.simple_vx(efficacy=0.5)
rv = ss.routine_vx(product=vx, prob=0.5, start_year=2021)
sim = ss.Sim(diseases=sir3, networks='random', interventions=rv, start=2020, stop=2023, n_agents=2000, verbose=0).run()
print('old API ran:', sim.results.sir.cum_infections[-1] if 'cum_infections' in sim.results.sir else list(sim.results.sir.keys())[:5])

print('\n=== 5. old API + new product mixed: routine_vx(product=ss.Vaccine(...))?')
try:
    sir3 = ss.SIR(beta=0.05, init_prev=0.01)
    rv = ss.routine_vx(product=ss.Vaccine(efficacy=0.5), prob=0.5, start_year=2021)
    sim = ss.Sim(diseases=sir3, networks='random', interventions=rv, start=2020, stop=2023, n_agents=2000, verbose=0).run()
    print('ran, cum_infections', sim.results.sir.cum_infections[-1])
except Exception as E:
    print('FAILS:', type(E).__name__, str(E)[:300])

print('\n=== 6. new delivery on an old (code) disease: ss.vaccinate on ss.SIR?')
try:
    sir3 = ss.SIR(beta=0.05, init_prev=0.01)
    sim = ss.Sim(diseases=sir3, networks='random', interventions=ss.vaccinate(ss.Vaccine(efficacy=0.5), coverage=0.5, years=[2021]), start=2020, stop=2023, n_agents=2000, verbose=0).run()
    print('ran')
except Exception as E:
    print('FAILS:', type(E).__name__, str(E)[:300])

print('\n=== 7. ss.Vx aliases: when does ss.Vx() return a Vaccine?')
print(type(ss.Vx(efficacy=0.9)).__name__, type(ss.Vx()).__name__ if True else '', end=' | ')
try:
    print(type(ss.Vx(diseases='sir')).__name__)
except Exception as E:
    print('Vx(diseases=) FAILS:', type(E).__name__, str(E)[:100])
try:
    print('Dx():', type(ss.Dx()).__name__)
except Exception as E:
    print('Dx() FAILS:', type(E).__name__, str(E)[:100])
