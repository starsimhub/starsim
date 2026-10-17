""" Why does ss.transfer(coverage=0.8) deliver to 8000 of 20000 and protect more than ss.vaccinate? """
import numpy as np
import starsim as ss
kw = dict(verbose=0, unit='day', total_pop=20_000, stop=30)
def make(**kwargs):
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> S', dur=10), init_prev=0.01, name='sis', **kwargs)

for where in ['disease', 'sim']:
    for adjust in [None, {'infection': [1.0, 0.1]}]:
        if where == 'disease':
            sis = make().stratify(vaccinated=[False, True], adjust=adjust) if adjust else make().stratify(vaccinated=[False, True])
            sim = ss.Sim(sis, interventions=ss.transfer(vaccinated=True, coverage=0.8, years=[10]), solver='ode', **kw)
        else:
            if adjust: continue
            sis = make()
            sim = ss.Sim(sis, stratify=dict(vaccinated=[False, True]), interventions=ss.transfer(vaccinated=True, coverage=0.8, years=[10]), solver='ode', **kw)
        sim.run()
        res = sim.results
        d = res[[m.name for m in sim.interventions()][0]]
        print(f'{where:8s} adjust={adjust}: delivered={d.delivered.sum():.0f} eligible={d.eligible[10]:.0f} coverage={d.coverage[10]:.2f}  n_S by group at 11: {np.asarray(res.sis.n_S.by("vaccinated"))[11]}  n_I[11]={res.sis.n_I[11]:.0f} n_I[29]={res.sis.n_I[29]:.0f}')

print('--- no transfer, no vaccine, baseline:')
sim = ss.Sim(make(), solver='ode', **kw).run()
print(f'n_I[11]={sim.results.sis.n_I[11]:.0f} n_I[29]={sim.results.sis.n_I[29]:.0f}')
print('--- vaccinate 0.8 efficacy 0.9:')
sim = ss.Sim(make(), interventions=ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.8, years=[10]), solver='ode', **kw).run()
print(f'n_I[11]={sim.results.sis.n_I[11]:.0f} n_I[29]={sim.results.sis.n_I[29]:.0f}')
