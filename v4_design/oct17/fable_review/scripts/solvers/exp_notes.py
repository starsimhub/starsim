"""Notes: Erlang stages, approx=, functions tried on groups, networks under ode, W1201/W1202, W102."""
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(total_pop=10_000, unit='day', stop=100, rand_seed=1, verbose=0.1)

def sir(dur=10, **extra):
    return ss.Disease(
        infection = ss.transmission('S -> I', beta=0.3),
        recovery  = ss.progression('I -> R', dur=dur),
        init_prev = 0.01, name='sir', **extra)

def run(label, *args, **kwargs):
    print(f'\n--- {label} ---')
    try:
        sim = ss.Sim(*args, **kwargs).init()
        print(f'  solver={sim.solver} reason={sim.solver_reason!r}')
        print(f'  notes={sim.notes}')
        return sim
    except Exception as e:
        print(f'  ERROR {type(e).__name__}: {str(e)[:400]}')

if False:
  pass
# A. Erlang notes (already run)
# run('weibull dwell, auto', sir(dur=ss.weibull(c=1.4, scale=ss.days(8))), **kw)
# run('lognormal dwell, auto', sir(dur=ss.lognorm_ex(mean=ss.days(10), std=ss.days(3))), **kw)
# run('lognormal dwell, auto, approx=True', sir(dur=ss.lognorm_ex(mean=ss.days(10), std=ss.days(3))), approx=True, **kw)
# run('lognormal dwell, auto, approx=False', sir(dur=ss.lognorm_ex(mean=ss.days(10), std=ss.days(3))), approx=False, **kw)
# run('lognormal dwell, ode, approx=False', sir(dur=ss.lognorm_ex(mean=ss.days(10), std=ss.days(3))), solver='ode', approx=False, **kw)
# run('lognormal more variable than exponential (std 20 > mean 10), auto', sir(dur=ss.lognorm_ex(mean=ss.days(10), std=ss.days(20))), **kw)
# run('weibull dwell, abm', sir(dur=ss.weibull(c=1.4, scale=ss.days(8))), solver='abm', **kw)
# run('gamma a=3 (Erlang), auto', sir(dur=ss.gamma(a=3, scale=ss.days(10/3))), **kw)
# run('weibull dwell with stages=5', ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=ss.weibull(c=1.4, scale=ss.days(8)), stages=5), init_prev=0.01, name='sir'), **kw)

# B. Functions tried on groups
def age_rate(sim):
    return 0.1 + 0.4*(sim.people.age > 65)
def partners(sim):
    return 0.1*sim.people.n_partners
def uids_fn(sim, uids):
    return np.full(len(uids), 0.1)
def scalar_fn(sim):
    return 0.1
def uses_I(sim):
    return 0.01*sim.diseases.sir.I.n/1000
sim = run('function of age in rate=, auto', ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', rate=age_rate), init_prev=0.01, name='sir'), **kw)
sim = run('function of age in rate=, auto, stratified by age', ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', rate=age_rate), init_prev=0.01, name='sir'), stratify=dict(age=[0, 20, 65]), **kw)
sim = run('function of n_partners, auto', ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', rate=partners), init_prev=0.01, name='sir'), **kw)
sim = run('function of n_partners, ode', ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', rate=partners), init_prev=0.01, name='sir'), solver='ode', **kw)
sim = run('function taking uids, auto', ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', rate=uids_fn), init_prev=0.01, name='sir'), **kw)
sim = run('scalar function, auto', ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', rate=scalar_fn), init_prev=0.01, name='sir'), **kw)
sim = run('function of I.n, auto', ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', rate=uses_I), init_prev=0.01, name='sir'), **kw)
if sim is not None:
    print('  solvers():'); sim.solvers()

# C. Networks under compartmental solvers
run('RandomNet, auto', sir(), networks=ss.RandomNet(n_contacts=10), **kw)
run('RandomNet, ode', sir(), networks=ss.RandomNet(n_contacts=10), solver='ode', **kw)
run('RandomNet, ode, approx=False', sir(), networks=ss.RandomNet(n_contacts=10), solver='ode', approx=False, **kw)
run('MFNet, ode', sir(), networks=ss.MFNet(), solver='ode', **kw)
run('MixingPool, auto', sir(), networks=ss.MixingPool(n_contacts=10), **kw)
run('infection_log analyzer, auto', sir(), analyzers='infection_log', **kw)
run('infection_log analyzer, ode', sir(), analyzers='infection_log', solver='ode', **kw)

# D. W1201/W1202, W102
print('\n--- small start, ode: W1201 ---')
s = ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init=dict(I=2), name='sir'), **kw).run()
print(f'  notes={s.notes}')
print('\n--- fast transition, binomial: W102 ---')
s = ss.Sim(sir(dur=2), solver='binomial', **kw).run()
print(f'  notes={s.notes}')
print('\n--- fast transition, abm: W102 ---')
s = ss.Sim(sir(dur=2), solver='abm', **(kw | dict(total_pop=2000))).run()
print(f'  notes={s.notes}')
print('\n--- fast transition, ode: no W102 ---')
s = ss.Sim(sir(dur=2), solver='ode', **kw).run()
print(f'  notes={s.notes}')
print('\n--- fast transition, gillespie ---')
s = ss.Sim(sir(dur=2), solver='gillespie', **kw).run()
print(f'  notes={s.notes}')
