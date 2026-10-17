"""Attaching data: CSV, mismatched names, aliases, long form, plotting"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import os
here = os.path.dirname(os.path.abspath(__file__))

def make():
    S, I, R = ss.states(3)
    infection = S >> I @ 0.3 * I
    recovery  = I >> R @ ss.days(10)
    cases = ss.outcome(0.5 * infection, delay=ss.days(2), every=ss.weeks(1))
    return ss.Disease(infection, recovery, cases, prevalence=I/(S+I+R), init_prev=0.02, name='sir')

kw = dict(total_pop=1000, start='2020-01-01', stop='2020-03-01', verbose=0)

# 1. Generate data from one run and write a CSV
sim = ss.Sim(make(), **kw).run()
res = sim.results.sir
weekly = res.cases.values[res.cases.values > 0]
dates = [str(d) for d in res.cases.timevec[res.cases.values > 0]]
df = pd.DataFrame(dict(date=dates, cases=weekly))
csv = os.path.join(here, 'cases.csv')
df.to_csv(csv, index=False)
print(df.head(3))

# 2. Attach by filename
sim = ss.Sim(make(), data=csv, **kw).run()
print('sim.data type', type(sim.data).__name__, 'cols', list(sim.data.columns), 'index type', type(sim.data.index).__name__)
cmp = ss.calibration.compare(sim)
print('compare keys', list(cmp.keys()), 'first pair', cmp['cases'][0][:3], cmp['cases'][1][:3])
fig = sim.plot()
print('sim.plot ok; scatter collections per axis:', [len(ax.collections) for ax in fig.axes])

# 3. Mismatched header
df_bad = df.rename(columns=dict(cases='Reported Cases'))
try:
    ss.Sim(make(), data=df_bad, **kw).init()
except Exception as e:
    print('mismatched ->', type(e).__name__, str(e)[:400])
# aliases
d = ss.data(df_bad, aliases=dict(cases='Reported Cases'))
sim = ss.Sim(make(), data=d, **kw).init()
print('aliased cols', list(sim.data.columns))
# ignore_extra
df_extra = df.assign(notes=['x']*len(df))
try:
    ss.Sim(make(), data=df_extra, **kw).init()
except Exception as e:
    print('extra col ->', type(e).__name__, str(e)[:300])
sim = ss.Sim(make(), data=ss.data(df_extra, ignore_extra=True), **kw).init()
print('ignore_extra cols', list(sim.data.columns))
# module-qualified name and underscore
sim = ss.Sim(make(), data=df.rename(columns=dict(cases='sir.cases')), **kw).init()
print('sir.cases ok', list(sim.data.columns))
sim = ss.Sim(make(), data=df.rename(columns=dict(cases='sir_cases')), **kw).init()
print('sir_cases ok', list(sim.data.columns))
# a Starsim-3 style result name
sim = ss.Sim(make(), data=df.rename(columns=dict(cases='new_infections')), **kw).init()
print('new_infections ok', list(sim.data.columns))
# the time column: 'Week ending' alias
d = ss.data(df.rename(columns={'date':'Week ending'}), aliases=dict(date='Week ending'))
sim = ss.Sim(make(), data=d, **kw).init()
print('date alias ok', sim.data.index[:2].tolist())
# unrecognised time column
try:
    ss.Sim(make(), data=df.rename(columns={'date':'week'}), **kw).init()
except Exception as e:
    print('no time col ->', type(e).__name__, str(e)[:300])
# data outside period
try:
    ss.Sim(make(), data=df.assign(date=[d.replace('2020','2023') for d in df.date]), **kw).run()
    print('data outside period: no error at run; compare ->', end=' ')
    try: ss.calibration.compare(sim)
    except Exception as e: print(type(e).__name__, str(e)[:200])
except Exception as e:
    print('data outside ->', type(e).__name__, str(e)[:300])

# 4. Long form by stratum
sirA = make().stratify(age=[0, 15, 65])
sim = ss.Sim(sirA, **kw).run()
r = sim.results.sir.cases
rows = []
for lvl in ['0-14', '15-64', '65+']:
    v = r.select(age=lvl).values
    for d, x in zip(r.timevec, v):
        if x > 0: rows.append(dict(date=str(d), age=lvl, cases=x))
long = pd.DataFrame(rows)
print(long.head(4))
sim2 = ss.Sim(make().stratify(age=[0, 15, 65]), data=long, **kw).run()
print('long -> wide cols', list(sim2.data.columns))
cmp = ss.calibration.compare(sim2)
print('compare keys', list(cmp.keys()))
fig = sim2.plot()
print('scatter per axis', [len(ax.collections) for ax in fig.axes], '(does the stratified data get drawn?)')
# a bad level label
try:
    ss.Sim(make().stratify(age=[0, 15, 65]), data=long.assign(age=long.age.replace({'65+':'65plus'})), **kw).init()
except Exception as e:
    print('bad level ->', type(e).__name__, str(e)[:300])
# long-form with a bin edge number instead of label
long2 = long.assign(age=long.age.map({'0-14':0, '15-64':15, '65+':65}))
try:
    s = ss.Sim(make().stratify(age=[0, 15, 65]), data=long2, **kw).init()
    print('numeric age column ->', list(s.data.columns))
except Exception as e:
    print('numeric age ->', type(e).__name__, str(e)[:300])

# 5. ss.data with a dataframe and a non-declared (Starsim 3) sim
s3 = ss.Sim(diseases='sis', networks='random', n_agents=500, start=2000, stop=2003, data=pd.DataFrame(dict(year=[2001, 2002], prevalence=[0.3, 0.4])), verbose=0)
try:
    s3.run(); print('ss3 data ok', list(s3.data.columns))
    try: print('compare on ss3:', ss.calibration.compare(s3))
    except Exception as e: print('compare on ss3 ERR', type(e).__name__, str(e)[:200])
except Exception as e:
    print('ss3 data ERR', type(e).__name__, str(e)[:300])
