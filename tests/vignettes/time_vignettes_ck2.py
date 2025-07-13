import starsim as ss


#%% Numerical start and stop #819
kw = dict(diseases='sis', networks='random')

s1 = ss.Sim(unit='month', dt=1.0, start=1980, stop=2000, **kw).run()
s2 = ss.Sim(unit='month', dt=1.0, start='1980-01-01', stop='2000-01-01', **kw).run()

#%% Durations interpreted as years #868
ss.Sim(start=2025, dur=12, unit='month').init()
# Out[12]: Sim(n=10,000; 2025—2037)

ss.Sim(start='2025-01-01', dur=12, unit='month').init()
# Out[15]: Sim(n=10,000; 2025-01-01—2026-01-01)

#%% Month vs year #917
import numpy as np
import sciris as sc
import starsim as ss

pars = dict(
    unit = 'year',
    dt = 1/12,
    start = 1990,
    stop = 1995,
)

sis = ss.SIS(unit='month', dt=1)


sim = ss.Sim(pars, diseases=sis, networks='random')
sim.run()

t = sc.objdict()
t.sim = sim.t.abstvec
t.sis = sim.diseases.sis.t.abstvec

months = list('jfmamjjasond')

diffs = sc.objdict()
for k,v in t.items():
    diffs[k] = np.diff(v[:13])*365

df = sc.dataframe(month=months, sim=diffs.sim, sis=diffs.sis)
print(df)