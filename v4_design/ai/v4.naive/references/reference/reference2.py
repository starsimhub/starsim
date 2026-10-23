"""Hidden reference answers for the second (harder) task set, SciPy/NumPy only."""
import numpy as np, pandas as pd
from scipy.integrate import solve_ivp
kw = dict(method='LSODA', rtol=1e-10, atol=1e-6, dense_output=True)
ans = {}
# H1: N=1e6, uniform mixing, 10% under 5; R0 2.5, D 7, 100 seeds on 2025-03-01; on 2025-06-01 80% of under-5s fully protected; infections to 2025-12-31 (excl seeds)
N, b, D = 1e6, 2.5/7, 7
f = lambda t, y: [-b*y[0]*y[1]/N, b*y[0]*y[1]/N - y[1]/D]
tv = (pd.Timestamp('2025-06-01') - pd.Timestamp('2025-03-01')).days
tend = (pd.Timestamp('2025-12-31') - pd.Timestamp('2025-03-01')).days
s1 = solve_ivp(f, [0, tv], [N-100, 100], **kw)
S, I = s1.y[:, -1]; inf1 = N-100-S
Sv = S - 0.8*0.1*S  # homogeneous mixing: S among under-5s is 10% of S
s2 = solve_ivp(f, [tv, tend], [Sv, I], **kw)
ans['H1 infections with campaign'] = inf1 + Sv - s2.y[0, -1]
s0 = solve_ivp(f, [0, tend], [N-100, 100], **kw)
ans['H1 infections without'] = N-100 - s0.y[0, -1]
ans['H1 averted'] = ans['H1 infections without'] - ans['H1 infections with campaign']
ans['H1 day of campaign'] = tv
# H2: SIR R0 2.5, D 7 (rate out of I 1/7), 2% of those leaving I die; N 1e5, 10 seeds, 365 d; deaths
for label, live in [('N = living', True), ('N fixed', False)]:
    def f(t, y):
        S, I, R, Dd = y; n = S+I+R if live else 1e5
        return [-b*S*I/n, b*S*I/n - I/7, 0.98*I/7, 0.02*I/7]
    s = solve_ivp(f, [0, 365], [1e5-10, 10, 0, 0], **kw)
    ans[f'H2 deaths ({label})'] = s.y[3, -1]
# H3: SIR R0 2, D 5, N 1e6, 10 seeds, start 2025-01-01: calendar date of the peak in daily new infections
b3 = 2/5
f = lambda t, y: [-b3*y[0]*y[1]/1e6, b3*y[0]*y[1]/1e6 - y[1]/5]
s = solve_ivp(f, [0, 365], [1e6-10, 10], **kw)
t = np.linspace(0, 365, 365001); S, I = s.sol(t)
inc_t = t[(b3*S*I).argmax()]
cum = 1e6 - 10 - s.sol(np.arange(366))[0]; daily = np.diff(cum)  # infections during day k -> k+1
ans['H3 instantaneous peak day'] = inc_t
ans['H3 peak day of daily totals (day k = [k, k+1))'] = daily.argmax()
ans['H3 date (instantaneous)'] = str((pd.Timestamp('2025-01-01') + pd.Timedelta(days=inc_t)).date())
# H4: stochastic SIR (exact Gillespie), N 1e4, R0 2, D 7, 10 seeds: 5th-95th percentile of final size (ever infected incl seeds)
rng = np.random.default_rng(7)
def run(N=10000, I0=10, R0=2.0):
    S, I = N - I0, I0
    while I > 0:
        pi = R0*S/N  # P(next event is infection) = beta S I/N / (beta S I /N + gamma I)
        pi = pi/(pi + 1)
        if rng.random() < pi: S -= 1; I += 1
        else: I -= 1
    return (N - S)/N
fs = np.array([run() for _ in range(400)])
ans['H4 5th percentile'] = np.percentile(fs, 5)
ans['H4 95th percentile'] = np.percentile(fs, 95)
ans['H4 median'] = np.median(fs)
for k, v in ans.items():
    print(f'{k}: {v}')
