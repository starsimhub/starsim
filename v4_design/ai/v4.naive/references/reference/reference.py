"""
Hidden reference answers for the naive-user tasks, computed without Starsim (SciPy only).
Run: /software/conda/envs/v4/bin/python reference.py
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from pathlib import Path

here = Path(__file__).parent
kw = dict(method='LSODA', rtol=1e-9, atol=1e-6, dense_output=True)
ans = {}

# T1: SIR final size, R0=2.5, D=7 days, N=1e5, 10 seeds, 365 days
N, I0, R0, D = 1e5, 10, 2.5, 7
b, g = R0/D, 1/D
f = lambda t, y: [-b*y[0]*y[1]/N, b*y[0]*y[1]/N - g*y[1], g*y[1]]
s = solve_ivp(f, [0, 365], [N-I0, I0, 0], **kw)
ans['T1 final size (fraction ever infected, incl. seeds)'] = (N - s.y[0, -1])/N

# T2: SEIR, R0=3, latent 5 d, infectious 7 d, N=1e6, 100 seeds infectious; peak day of I
N, R0, L, D = 1e6, 3, 5, 7
b = R0/D
f = lambda t, y: [-b*y[0]*y[2]/N, b*y[0]*y[2]/N - y[1]/L, y[1]/L - y[2]/D, y[2]/D]
s = solve_ivp(f, [0, 365], [N-100, 0, 100, 0], **kw)
t = np.linspace(0, 365, 365001); I = s.sol(t)[2]
ans['T2 peak day of infectious'] = t[I.argmax()]
ans['T2 peak infectious count'] = I.max()

# T3: SIRS, R0=3, infectious 5 d, immunity mean 1 year; endemic prevalence (fraction I) after 20 years
R0, D, W = 3, 5, 365
g, w = 1/D, 1/W
ans['T3 endemic prevalence (analytic)'] = (1 - 1/R0)*w/(w + g)

# T4: SIR with births=deaths, life expectancy 50 y, R0=10, infectious 14 d; equilibrium fraction susceptible
ans['T4 equilibrium fraction susceptible (analytic, R0 incl. mortality ~ 1/R0)'] = 1/10

# T5: two age groups, children 30%, adults 70%; contacts per day C=[[10,3],[2,6]] (row=person's group);
# beta per contact 0.02, infectious 7 days, N=1e6, 0.01% infected in each group; attack rates by day 365
N = 1e6; Ns = np.array([0.3, 0.7])*N; C = np.array([[10, 3], [2, 6]]); beta = 0.02; D = 7
def f(t, y):
    S, I = y[:2], y[2:4]
    lam = beta*C @ (I/Ns)
    return np.concatenate([-lam*S, lam*S - I/D, I/D])
y0 = np.concatenate([Ns*(1-1e-4), Ns*1e-4, [0, 0]])
s = solve_ivp(f, [0, 365], y0, **kw)
ans['T5 attack rate children'] = 1 - s.y[0, -1]/Ns[0]
ans['T5 attack rate adults'] = 1 - s.y[1, -1]/Ns[1]
ans['T5 R0 (beta*D*rho(C))'] = beta*D*max(abs(np.linalg.eigvals(C)))

# T6: SIR R0=2, D=7, N=1e6, 100 seeds; 30% vaccinated on day 0 with all-or-nothing 90% efficacy; total infections by day 365
N, R0, D = 1e6, 2, 7; b = R0/D
p = 0.3*0.9
f = lambda t, y: [-b*y[0]*y[1]/N, b*y[0]*y[1]/N - y[1]/D]
S0 = (N-100)*(1-p)
s = solve_ivp(f, [0, 365], [S0, 100], **kw)
ans['T6 infections (excluding the 100 seeds)'] = S0 - s.y[0, -1]
s0 = solve_ivp(f, [0, 365], [N-100, 100], **kw)
ans['T6 infections with no vaccine (excluding seeds)'] = (N-100) - s0.y[0, -1]
ans['T6 infections averted'] = ans['T6 infections with no vaccine (excluding seeds)'] - ans['T6 infections (excluding the 100 seeds)']

# T7: SIR R0=2.5, D=7, N=1e6, 100 seeds; transmission cut 40% from day 30 to day 90; infections by day 365
N, R0, D = 1e6, 2.5, 7
def f(t, y):
    b = R0/D*(0.6 if 30 <= t < 90 else 1)
    return [-b*y[0]*y[1]/N, b*y[0]*y[1]/N - y[1]/D]
s = solve_ivp(f, [0, 365], [N-100, 100], max_step=0.1, **kw)
ans['T7 infections with closure (excl seeds)'] = (N-100) - s.y[0, -1]
s0 = solve_ivp(lambda t, y: [-R0/D*y[0]*y[1]/N, R0/D*y[0]*y[1]/N - y[1]/D], [0, 365], [N-100, 100], **kw)
ans['T7 infections no closure (excl seeds)'] = (N-100) - s0.y[0, -1]

# T8: weekly-case data from an SIR with R0=1.8 (beta=0.3/day, D=6 d), N=1e5, 20 seeds, 30% of infections reported; true R0
N, b, D = 1e5, 0.3, 6
f = lambda t, y: [-b*y[0]*y[1]/N, b*y[0]*y[1]/N - y[1]/D]
s = solve_ivp(f, [0, 140], [N-20, 20], **kw)
cum = N - 20 - s.sol(np.arange(0, 141, 7))[0]
weekly = np.diff(cum)*0.3
cases = np.random.default_rng(42).poisson(weekly)
dates = pd.date_range('2025-01-06', periods=20, freq='7D') + pd.Timedelta(days=7)
pd.DataFrame(dict(week_ending=dates.date, cases=cases)).to_csv(here/'t8_weekly_cases.csv', index=False)
ans['T8 true R0'] = b*D

# T9: probability a single case causes a major outbreak, R0=2 (exponential infectious period)
ans['T9 P(major outbreak)'] = 1 - 1/2

# T10: SIR with beta 1.5 per week, recovery mean 2 weeks, N=1e4, 10 seeds, 1 year with a daily step; final size
N, b, D = 1e4, 1.5/7, 14
f = lambda t, y: [-b*y[0]*y[1]/N, b*y[0]*y[1]/N - y[1]/D]
s = solve_ivp(f, [0, 365], [N-10, 10], **kw)
ans['T10 final size fraction'] = (N - s.y[0, -1])/N

for k, v in ans.items():
    print(f'{k}: {v:.6g}')
