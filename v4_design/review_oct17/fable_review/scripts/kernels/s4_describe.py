"""describe() and the loop for a declared SIR with a delivery, as an ODE and as agents."""
import numpy as np
import sciris as sc
import starsim as ss

def make():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=8), init=dict(I=0.01), name='sir')

for solver in ['ode', 'abm', 'binomial']:
    print(f'\n################ solver={solver} ################')
    vax = ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.5, start=20, stop=21)
    kw = dict(unit='day', stop=60, total_pop=5000, verbose=0, solver=solver)
    if solver == 'abm':
        kw['n_agents'] = 2000
    sim = ss.Sim(make(), interventions=vax, **kw).init()
    sim.describe()
    print('kernel_only:', sim.kernel_only(), '|', sim.kernel_reason())
    print('tabulated:', getattr(sim.interventions[0], 'tabulated', None))
    df = sim.loop.to_df()
    one = df[df.ti == 2]
    print('loop at ti=2:', list(one.label))
    sim.run()
    r = sim.results
    print('delivered total:', float(np.sum(r[vax.name].delivered.values)) if vax.name in r else 'n/a', 'n_I max', float(r.sir.n_I.max()))

print('\n################ describe of a plain SIR run with stochastic solver: is CRN mentioned? ################')
sim = ss.Sim(make(), unit='day', stop=60, total_pop=5000, verbose=0, solver='binomial').init()
txt = sim.describe(output=True)
print([line for line in txt.split('\n') if 'random' in line.lower() or 'crn' in line.lower() or 'seed' in line.lower()] or 'no line about randomness/CRN/seed in describe()')
print('\n################ loop.insert on a declared model that was kernel_only ################')
sim = ss.Sim(make(), unit='day', stop=60, total_pop=5000, verbose=0).init()
print('before insert kernel_only:', sim.kernel_only())
sim.loop.insert(lambda sim: None, label='sir.step')
print('after insert kernel_only:', sim.kernel_only(), '|', sim.kernel_reason())
