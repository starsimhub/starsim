"""Attempt B: guessing the v4 way from the overview page only."""
import traceback, starsim as ss, numpy as np, pandas as pd, pylab as pl

def attempt(label, fn):
    print(f'\n######## {label}')
    try: out = fn(); print('[ok]'); return out
    except Exception: print('\n'.join(traceback.format_exc().splitlines()[-5:])); print('[FAIL]')

def make_b1():
    S, E, I, R, D = ss.states(5)
    infection = S >> E @ 0.3 * I
    latent    = E >> I @ ss.days(5)
    outcome   = I >> (0.9*R + 0.1*D) @ ss.days(10)
    seir = ss.Disease(infection, latent, outcome, init_prev=0.01)
    return seir
def b1():
    seir = make_b1()
    sim = ss.Sim(seir, start='2020-01-01', stop='2022-01-01', total_pop=1e6, verbose=0).run()
    sim.describe()
    print(sim.summarize())
attempt('B1 operators, weighted destination', b1)

def b2():
    seir = ss.Disease(
        infection = ss.transmission('S -> E', beta=0.3, source='I'),
        latent    = ss.progression('E -> I', dur=5),
        outcome   = ss.progression('I -> R | D', dur=10, p=[0.9, 0.1]),
        init_prev = 0.01,
    )
    sim = ss.Sim(seir, start='2020-01-01', stop='2022-01-01', total_pop=1e6, verbose=0).run()
    sim.describe()
attempt('B2 strings, branching', b2)

def b3():
    S, E, I, R = ss.states(4)
    seir = ss.Disease(S >> E @ 0.3*I, E >> I @ ss.days(5), I >> (0.9*R + 0.1*ss.null) @ ss.days(10), init_prev=0.01)
    ss.Sim(seir, start='2020-01-01', stop='2022-01-01', total_pop=1e6, verbose=0).run().describe()
attempt('B3 death to ss.null in a weighted destination', b3)

def b3b():
    S, E, I, R = ss.states(4)
    infection = S >> E @ 0.3*I
    latent    = E >> I @ ss.days(5)
    recovery  = I >> R @ ss.days(10)
    death     = ss.death('I ->', p=0.1)
    seir = ss.Disease(infection, latent, recovery, death, init_prev=0.01)
    ss.Sim(seir, start='2020-01-01', stop='2022-01-01', total_pop=1e6, verbose=0).run().describe()
attempt('B3b ss.death with p=', b3b)

def v1():
    seir = make_b1()
    vx = ss.vaccinate(ss.Vaccine(efficacy=0.8), coverage=0.8, years=[2021], eligible=lambda sim: sim.people.age > 65)
    sim = ss.Sim(seir, interventions=vx, start='2020-01-01', stop='2022-01-01', total_pop=1e6, verbose=0).run()
    sim.describe()
attempt('V1 vaccinate with lambda eligibility, no stratification', v1)

def v2():
    seir = make_b1()
    ppl = ss.People(1)
    vx = ss.vaccinate(ss.Vaccine(efficacy=0.8), coverage=0.8, years=[2021], eligible=ppl.age > 65)
    sim = ss.Sim(seir, interventions=vx, start='2020-01-01', stop='2022-01-01', total_pop=1e6, stratify=dict(age=[0, 65]), verbose=0).run()
    sim.describe()
    print(sim.results.keys())
attempt('V2 vaccinate with lazy condition and age stratification', v2)

def v3():
    seir = make_b1()
    ppl = ss.People(1)
    seir.stratify(vaccinated=[False, True], adjust={'infection': [1.0, 0.2]})
    rollout = ss.transfer(vaccinated=True, coverage=0.8, years=[2021], eligible=ppl.age > 65)
    sim = ss.Sim(seir, interventions=rollout, start='2020-01-01', stop='2022-01-01', total_pop=1e6, stratify=dict(age=[0, 65]), verbose=0).run()
    sim.describe()
attempt('V3 vaccination as stratification + transfer', v3)
