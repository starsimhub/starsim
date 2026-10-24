""" A vaccine with waning, under ode and abm """
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(verbose=0, unit='day', total_pop=20_000, stop=120)

def make():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> S', dur=10), init_prev=0.01, name='sis')

print('=== 1. waning as a duration (exponential loss) vs as a function (fades)')
for label, waning in [('duration', ss.days(30)), ('function', lambda t: np.exp(-t/30)), ('none', None)]:
    for solver in ['ode', 'abm']:
        sis = make()
        bcg = ss.Vaccine(efficacy=0.9, waning=waning)
        rounds = ss.vaccinate(bcg, coverage=0.8, years=[10])
        sim = ss.Sim(sis, interventions=rounds, solver=solver, n_agents=2000, **kw)
        sim.run()
        print(f'{label:9s} {solver:4s} n_I[60]={sim.results.sis.n_I[60]:8.0f}  strats={[s.name for s in sim.strats]}  levels={[len(s.labels) for s in sim.strats]}')

print('\n=== 2. describe() and what_modifies for the vaccine')
sis = make()
bcg = ss.Vaccine(efficacy=0.9, waning=ss.days(30), cost=1.5)
rounds = ss.vaccinate(bcg, coverage=0.8, years=[10])
sim = ss.Sim(sis, interventions=rounds, solver='abm', n_agents=2000, **kw)
sim.init()
print(bcg.describe())
print(rounds.describe())
sim.what_modifies(sis.rel_sus)
sim.run()
print('results of the delivery:', list(sim.results.rounds.keys()))
print('product is a module in the sim?', bcg.name, 'in', [m.name for m in sim.modules])
print('has_bcg property:', type(sim.people['has_bcg']).__name__)

print('\n=== 3. what does the spec say about waning as a duration? "exponential decay with that mean". Does the ABM fade efficacy or drop it?')
sis = make()
bcg = ss.Vaccine(efficacy=0.9, waning=ss.days(30))
rounds = ss.vaccinate(bcg, coverage=1.0, years=[10])
sim = ss.Sim(sis, interventions=rounds, solver='abm', n_agents=2000, **kw).run()
eff = sim.diseases.sis.rel_sus.effective.raw
held = sim.people['has_bcg'].raw
print('unique effective rel_sus at the end:', np.unique(np.round(eff, 3)))
print('share still holding at t=120 (expected exp(-110/30)=%.3f):' % np.exp(-110/30), held.mean())

print('\n=== 4. the product name from the variable; two deliveries of one product; a product with no variable')
sis = make()
v = ss.Vaccine(efficacy=0.5)
a = ss.vaccinate(v, coverage=0.5, years=[10])
b = ss.vaccinate(v, coverage=0.5, years=[20])
sim = ss.Sim(sis, interventions=[a, b], solver='ode', **kw).run()
print('strats:', [s.name for s in sim.strats], '| deliveries:', [m.name for m in sim.interventions()])
sis = make()
sim = ss.Sim(sis, interventions=ss.vaccinate(ss.Vaccine(efficacy=0.5), coverage=0.5, years=[10]), solver='ode', **kw).run()
print('inline: strats:', [s.name for s in sim.strats], '| deliveries:', [m.name for m in sim.interventions()])

print('\n=== 5. a vaccine reused in two sims (scenarios): does the state leak?')
sis = make()
v = ss.Vaccine(efficacy=0.5)
s1 = ss.Sim(sis, interventions=ss.vaccinate(v, coverage=0.5, years=[10]), solver='abm', n_agents=2000, **kw).run()
s2 = ss.Sim(make(), interventions=ss.vaccinate(v, coverage=0.5, years=[10]), solver='abm', n_agents=2000, **kw).run()
print('s1', s1.results.sis.n_I[-1], 's2', s2.results.sis.n_I[-1])

print('\n=== 6. vaccination as a transfer vs as a product (spec: "One should be the documented way")')
sis = make().stratify(vaccinated=[False, True], adjust={'infection': [1.0, 0.1]})
rollout = ss.transfer(vaccinated=True, coverage=0.8, years=[10])
s1 = ss.Sim(sis, interventions=rollout, solver='ode', **kw).run()
sis = make()
s2 = ss.Sim(sis, interventions=ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.8, years=[10]), solver='ode', **kw).run()
print('transfer n_I[60]:', s1.results.sis.n_I[60], ' vaccinate n_I[60]:', s2.results.sis.n_I[60])
