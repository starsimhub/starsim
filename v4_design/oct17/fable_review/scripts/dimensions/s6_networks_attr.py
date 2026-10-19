import starsim as ss
home, school = ss.contacts('Kenya', layer='home'), ss.contacts('Kenya', layer='school')
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=ss.days(10)), init_prev=0.01, name='sir')
sim = ss.Sim(sir, networks=[home, school], stratify=dict(age=[0, 5, 18, 65]), unit='day', stop=5, verbose=0).init()
print('sim.networks keys:', list(sim.networks.keys()), '| pars.networks keys:', list(sim.pars.networks.keys()))
print('hasattr school:', hasattr(sim.networks, 'school'))
