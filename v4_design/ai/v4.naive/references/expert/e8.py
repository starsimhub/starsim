import numpy as np, pandas as pd, starsim as ss
df = pd.read_csv('../reference/t8_weekly_cases.csv').rename(columns=dict(week_ending='date'))
S, I, R = ss.states(3)
beta = ss.par(0.2, bounds=[0.05, 1.0])
infection = S >> I @ beta * I
recovery = I >> R @ ss.days(6)
cases = ss.outcome(0.3 * infection)
sir = ss.Disease(infection, recovery, cases, init=dict(I=20), name='sir')
sim = ss.Sim(sir, data=df, total_pop=1e5, start='2025-01-06', stop='2025-06-01')
fit = sim.calibrate()
print(fit)
b = fit.sim.diseases.sir.pars.beta
print('beta', b, 'R0', float(getattr(b,'value',b))*6)
