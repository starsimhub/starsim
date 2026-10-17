import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import numpy as np
import starsim as ss

def make(name='sir', **kwargs):
    return ss.Disease(recovery=ss.progression('I -> R',rate=.2), init=dict(I=100),name=name,**kwargs)
def sim(**kwargs):
    return ss.Sim(make(), solver='binomial',unit='day',stop=8,total_pop=100,verbose=0,**kwargs)
for profile in [False,True]:
    s=sim().run(profile=profile)
    print('profile',profile,s.results.sir.n_I.values)
for names in [['ab'],['ab','ba'],['ba','ab']]:
    s=ss.Sim([make(n) for n in names],solver='binomial',unit='day',stop=8,total_pop=100,verbose=0).run(profile=True)
    print('names',names,[(d.name,d.results.n_I.values.tolist()) for d in s.declared])
s=ss.Sim([make('ab'),make('ba')],solver='binomial',unit='day',stop=8,total_pop=100,verbose=0).run()
print('whole-run colliding names', [(d.name,d.results.n_I.values.tolist()) for d in s.declared])
try:
    s=ss.Sim(make(dt=ss.days(2)),solver='ode',unit='day',dt=1,stop=8,total_pop=100,verbose=0)
    print('check_dt',s.check_dt(output=True))
except Exception as e:
    print('check_dt error',type(e).__name__,str(e))
for stop in [1800,2000]:
    try:
        s=ss.Sim(make(),stop=stop,total_pop=100,verbose=0).init()
        print('stop',stop,'unit',s.unit,'timeline',s.t.start,s.t.stop,'dt',s.t.dt)
    except Exception as e: print('stop',stop,type(e).__name__,str(e))
for crn in [True,False]:
    with ss.options.context(crn=crn):
        print('crn',crn,sim().hash())

hashes = []
for multiplier in [0.1, 0.9]:
    d = ss.Disease(infection=ss.transmission('S -> I', beta=.1),
                   recovery=ss.progression('I -> R', dur=10), init_prev=.1, name='sir')
    effect = ss.modify(d.pars.beta, multiply=multiplier, name='policy')
    s = ss.Sim(d, connectors=effect, solver='ode', unit='day', total_pop=100,
               start=0, stop=2, verbose=0)
    hashes.append(s.hash())
    s.run()
    print('connector multiplier', multiplier, 'hash', hashes[-1],
          'final infected', s.results.sir.n_I.values[-1])
print('Connector changes leave all hash components equal:', hashes[0] == hashes[1])
