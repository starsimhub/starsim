"""Routes: beta per route and direction, E403, spec examples that should run"""
import numpy as np
import sciris as sc
import starsim as ss

print('== A. beta per network and per direction, as agents')
try:
    hiv = ss.Disease(infection=ss.transmission('S -> I', beta=dict(mf=[0.008, 0.004], needles=0.03)), recovery=ss.progression('I -> R', dur=ss.years(10)), init_prev=0.05, name='hiv')
    needles = ss.RandomNet(n_contacts=2, name='needles')
    sim = ss.Sim(hiv, networks=[ss.MFNet(), needles], n_agents=2000, start=2000, stop=2003, verbose=0).run()
    print('ran; cum infections', float(sim.results.hiv.cum_infection[-1]))
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:300])

print('== B. a beta key that matches no route -> E403?')
try:
    hiv = ss.Disease(infection=ss.transmission('S -> I', beta=dict(mf=0.008, needls=0.03)), recovery=ss.progression('I -> R', dur=ss.years(10)), init_prev=0.05, name='hiv')
    ss.Sim(hiv, networks=[ss.MFNet(), ss.RandomNet(n_contacts=2, name='needles')], n_agents=1000, start=2000, stop=2001, verbose=0).init()
    print('no error')
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:300])

print('== C. spec 06 four-line example')
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=ss.days(10)), init_prev=0.01, name='sir')
for label, kw in [('no network', {}), ('MixingPool(n_contacts=10)', dict(networks=ss.MixingPool(n_contacts=10))), ('contacts(Kenya)', dict(networks=ss.contacts('Kenya'))), ('RandomNet(10), 2000 agents', dict(networks=ss.RandomNet(n_contacts=10), n_agents=2000))]:
    try:
        sim = ss.Sim(sc.dcp(sir), unit='day', stop=60, verbose=0, **kw).run()
        print(f'{label:28s} solver={sim.solver:4s} cum_infection={float(sim.results.sir.cum_infection[-1]):.0f} of {sim.pars.total_pop or sim.pars.n_agents}')
    except Exception as e:
        print(label, 'ERR', type(e).__name__, str(e)[:300])

print('== D. ss.contacts("Kenya") with no age stratification: what happens?')
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=ss.days(10)), init_prev=0.01, name='sir')
sim = ss.Sim(sir, networks=ss.contacts('Kenya'), unit='day', stop=10, verbose=0).init()
print('dims:', [(s.name, s.n) for s in sim.strats])

print('== E. ss.StaticNet under ode; ClusterNet under ode')
for net in [ss.StaticNet(n_contacts=10), ss.ClusterNet(cluster_size=4)]:
    try:
        sim = ss.Sim(sc.dcp(sir), networks=net, solver='ode', unit='day', stop=10, verbose=0).init()
        print(type(net).__name__, 'ok')
    except Exception as e:
        print(type(net).__name__, 'ERR', type(e).__name__, str(e)[:200])

print('== F. layers: contacts(Kenya, layer=home/school) and a modifier on the school layer')
home, school = ss.contacts('Kenya', layer='home'), ss.contacts('Kenya', layer='school')
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=ss.days(10)), init_prev=0.01, name='sir')
try:
    closure = ss.modify('school.pars.n_contacts', multiply=0.2, start=10, stop=40)
    sim = ss.Sim(sir, networks=[home, school], stratify=dict(age=[0, 5, 18, 65]), unit='day', stop=60, interventions=closure, verbose=0).run()
    print('ran; names:', list(sim.networks.keys()), 'cum', float(sim.results.sir.cum_infection[-1]))
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:300])
try:
    closure = ss.modify(school.pars.n_contacts, multiply=0.2, start=10, stop=40)  # the spec's spelling
    print('spec spelling accepted:', closure)
except Exception as e:
    print('spec spelling ERR', type(e).__name__, str(e)[:200])

print('== G. MixingPool(n_contacts=10) vs no network: beta meaning')
for kw in [{}, dict(networks=ss.MixingPool(n_contacts=1))]:
    sim = ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir'), stop=60, verbose=0, **kw).run()
    print('networks' in kw, float(sim.results.sir.cum_infection[-1]))
