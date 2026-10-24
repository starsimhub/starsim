"""Attempt A: a Starsim-3 modeler's first instinct, unchanged from v3."""
import traceback, starsim as ss, numpy as np, pandas as pd, pylab as pl

def attempt(label, fn):
    print(f'\n######## {label}')
    try: fn(); print('[ok]')
    except Exception: print('\n'.join(traceback.format_exc().splitlines()[-6:])); print('[FAIL]')

def a1():
    seir = ss.SEIR(pars=dict(dur_exp=ss.days(5), dur_inf=ss.days(10), beta=ss.perday(0.1), init_prev=0.01))
    net = ss.RandomNet(pars=dict(n_contacts=10))
    vx = ss.routine_vx(product=ss.Vx(pars=dict(efficacy=0.8)), prob=0.8, start_year=2021, end_year=2022,
                       eligibility=lambda sim: sim.people.age > 65)
    sim = ss.Sim(diseases=seir, networks=net, interventions=vx, start='2020-01-01', stop='2022-01-01', n_agents=1000, dt=ss.days(1))
    sim.run()
    print(sim.results.seir.keys())
    sim.plot(); pl.savefig('s5_a1.png')
attempt('v3 SEIR + RandomNet + routine_vx', a1)

def a2():
    seir = ss.SEIR(pars=dict(dur_exp=ss.days(5), dur_inf=ss.days(10), beta=ss.perday(0.1), init_prev=0.01))
    sim = ss.Sim(diseases=seir, networks=ss.RandomNet(n_contacts=10), start='2020-01-01', stop='2022-01-01', n_agents=1000, solver='ode')
    sim.run()
attempt('v3 SEIR as ODE', a2)

def a3():
    seir = ss.SEIR(pars=dict(dur_exp=ss.days(5), dur_inf=ss.days(10), beta=ss.perday(0.1), init_prev=0.01))
    sim = ss.Sim(diseases=seir, start='2020-01-01', stop='2022-01-01', solver='ode')
    sim.run()
    sim.describe()
attempt('v3 SEIR as ODE, no network', a3)

def a4():
    # v3 modeler's way to add case fatality: subclass
    class SEIRD(ss.SEIR):
        def __init__(self, pars=None, **kwargs):
            super().__init__(**kwargs)
            self.define_pars(p_death=ss.bernoulli(0.1))
            self.update_pars(pars, **kwargs)
        def step_state(self):
            super().step_state()
    seird = SEIRD(pars=dict(dur_exp=ss.days(5), dur_inf=ss.days(10), beta=ss.perday(0.1), init_prev=0.01))
    sim = ss.Sim(diseases=seird, networks=ss.RandomNet(n_contacts=10), stop='2021-01-01', start='2020-01-01', n_agents=1000, verbose=0).run()
    print(sim.summarize())
attempt('v3 subclass', a4)
