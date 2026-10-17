"""Misc checks: so_far, resample on numeric time, list-form results=, every= vs daily data, by() access"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

def make(beta=0.3, **extra):
    S, I, R = ss.states(3)
    infection = S >> I @ beta * I
    recovery  = I >> R @ ss.days(10)
    return ss.Disease(infection, recovery, init=dict(I=3), name='sir', **extra)

# 1. Reading a result during the run
class Watch(ss.Analyzer):
    compartmental = True
    def step(self):
        r = self.sim.results.sir.infection
        if self.ti == 5:
            print('1. at ti=5: infection[:ti]', np.round(r[:self.ti], 2), ' len', len(r), ' has so_far?', hasattr(r, 'so_far'), ' r[-3:] (wrong: future zeros)', r[-3:])
sim = ss.Sim(make(), analyzers=Watch(), solver='binomial', total_pop=300, unit='day', stop=20, verbose=0).run()

# 2. resample on a sim with unit=day and no dates, and a Starsim 3 one
try:
    sim.results.sir.infection.resample('week')
    print('2. declared, unit=day, no dates: resample ok')
except Exception as e:
    print('2. declared unit=day resample ERR', type(e).__name__, str(e)[:80])
s3 = ss.Sim(diseases='sis', networks='random', n_agents=200, unit='day', dur=20, verbose=0).run()
try:
    s3.results.sis.new_infections.resample('week'); print('2. ss3 unit=day resample ok')
except Exception as e:
    print('2. ss3 unit=day resample ERR', type(e).__name__, str(e)[:80])
s3 = ss.Sim(diseases='sis', networks='random', n_agents=200, start=2000, stop=2002, verbose=0).run()
print('2. ss3 start=2000 resample year', s3.results.sis.new_infections.resample('year').values)

# 3. The list form results=/outcomes=
S, I, R = ss.states(3)
N = S + I + R
infection = S >> I @ 0.3 * I
recovery  = I >> R @ ss.days(10)
prevalence = I / N
ever = (I + R) / N
cases = ss.outcome(0.5 * infection, delay=ss.days(2))
sir = ss.Disease(transitions=[infection, recovery], results=[prevalence, ever], outcomes=[cases], init=dict(I=3))
sim = ss.Sim(sir, total_pop=300, unit='day', stop=20, verbose=0).run()
print('3. list form keys', [k for k in sim.results[sir.name].keys() if k != 'timevec'], 'module name', sir.name)
# a sum of states as a result: N
S2, I2, R2 = ss.states(3)
sir2 = ss.Disease(S2 >> I2 @ 0.3*I2, I2 >> R2 @ ss.days(10), N=S2+I2+R2, init=dict(I2=3), name='sir2')
sim = ss.Sim(sir2, total_pop=300, unit='day', stop=5, verbose=0).run()
print('3. N as result:', sim.results.sir2.N.values[:3], 'I.frac exists?', hasattr(I2, 'frac'))

# 4. every= weekly outcome vs daily data
S, I, R = ss.states(3)
infection = S >> I @ 0.3 * I
recovery  = I >> R @ ss.days(10)
weekly = ss.outcome(0.5 * infection, every=ss.weeks(1))
daily = ss.outcome(0.5 * infection)
sir = ss.Disease(infection, recovery, weekly, daily, init=dict(I=3), name='sir')
sim = ss.Sim(sir, total_pop=300, start='2020-01-01', stop='2020-02-01', verbose=0).run()
r = sim.results.sir
print('4. weekly[:9]', np.round(r.weekly.values[:9], 1))
print('4. daily.resample(week)[:3]', np.round(r.daily.resample('week').values[:3], 1), ' weekly nonzero[:3]', np.round(r.weekly.values[r.weekly.values > 0][:3], 1))
dfw = pd.DataFrame(dict(date=[str(d) for d in r.timevec[7::7]], weekly=r.daily.values.reshape(-1)[1:][:28].reshape(4, 7).sum(axis=1) if False else [r.daily.values[i-6:i+1].sum() for i in range(7, 32, 7)]))
# daily data against the weekly outcome
dfd = pd.DataFrame(dict(date=[str(d) for d in r.timevec[1:8]], weekly=r.daily.values[1:8]))
sim2 = ss.Sim(sir, total_pop=300, start='2020-01-01', stop='2020-02-01', verbose=0, data=dfd).run()
cmp = ss.calibration.compare(sim2)
print('4. daily data vs every=week outcome -> model values', np.round(cmp['weekly'][1], 1), '(data', np.round(cmp['weekly'][0], 1), ')')
# weekly data against the daily outcome: summed over the interval
dfw = pd.DataFrame(dict(date=[str(d) for d in r.timevec[7::7]], daily=[r.daily.values[i-6:i+1].sum() for i in range(7, 32, 7)]))
sim3 = ss.Sim(sir, total_pop=300, start='2020-01-01', stop='2020-02-01', verbose=0, data=dfw).run()
cmp = ss.calibration.compare(sim3)
print('4. weekly data vs daily outcome -> model', np.round(cmp['daily'][1], 1), 'data', np.round(cmp['daily'][0], 1))

# 5. by() result access
sir = make().stratify(age=[0, 15, 65])
sim = ss.Sim(sir, total_pop=300, unit='day', stop=5, verbose=0).run()
b = sim.results.sir.n_I.by('age')
print('5. by cols', b.columns, " b['0-14'] type", type(b['0-14']).__name__, ' b.col ok', b.col('65+').name, ' attr access b.__dict__ keys subset', [k for k in ['columns','strata','strats','dims'] if k in b.__dict__])
try:
    print('5. b.select(age=0)?', b.select(age=0))
except Exception as e:
    print('5. by().select ERR:', str(e)[:80])
print('5. select no args:', sim.results.sir.n_I.select().label, ' == total?', np.allclose(sim.results.sir.n_I.select(), sim.results.sir.n_I))
print('5. select list:', sim.results.sir.n_I.select(age=[0, 1]).label)
print('5. select(age=0).select(age=1) (should be zero):', sim.results.sir.n_I.select(age=0).select(age=1).values[-1])
print('5. ss.Result has .sum?', hasattr(sim.results.sir.n_I, 'sum'), 'repr of selected:', str(sim.results.sir.n_I.select(age=0))[:60])
# 2D (columns) result from ss3 multi-strain? only check by() interplay: a stratified result's strata is kept but columns is None
print('5. n_I.columns', sim.results.sir.n_I.columns, 'n_I.strata.shape', sim.results.sir.n_I.strata.shape, 'strats', [s.name for s in sim.results.sir.n_I.strats], 'dims', sim.results.sir.n_I.dims)
# 6. names: cum_ for outcomes? label of transitions
print('6. labels', sim.results.sir.infection.label, sim.results.sir.n_I.label, sim.results.sir.infection.full_label)
# 7. sim.summarize ignores summarize_by
print('7. summarize infection (mean):', round(sim.summarize()['sir_infection'], 2), 'sum', round(sim.results.sir.infection.sum(), 2))
# 8. the module default name when no variable
d = ss.Disease(infection=ss.transmission('S -> I', beta=0.1), recovery=ss.progression('I -> R', dur=5))
print('8. default name', d.name)
