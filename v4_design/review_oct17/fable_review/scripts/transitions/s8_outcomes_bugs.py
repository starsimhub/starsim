"""Outcomes, and a few operator edge cases."""
import starsim as ss, numpy as np, sciris as sc

S, I, R = ss.states('S I R')
infection = S >> I @ 0.3*I
recovery  = I >> R @ ss.days(10)
cases = ss.outcome(0.5 * infection, delay=ss.days(5))
hosp  = ss.outcome(0.05 * infection, delay=ss.days(7), duration=ss.days(10))
weekly = ss.outcome(infection, every=ss.weeks(1))
d = ss.Disease(infection, recovery, cases, hosp, weekly, prevalence=I/(S+I+R), incidence=infection/S, init_prev=0.01, name='sir')
sim = ss.Sim(d, unit='day', stop=60, verbose=0).run()
res = sim.results.sir
print('results:', list(res.keys()))
print('cases[20]/infection[15]:', res.cases[20], res.infection[15])
print('n_hosp max:', res.n_hosp.max())
print('weekly[7], sum infection[1:8]:', res.weekly[7], res.infection[1:8].sum(), '| weekly[6]:', res.weekly[6])

print('\n--- an unnamed outcome passed positionally without a variable')
try:
    S, I, R = ss.states('S I R')
    ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), ss.outcome(0.5 * (S >> I @ 0.3*I)))
except Exception as e: print('ERROR', str(e)[:200])

print('\n--- recovery + death as an outcome (FlowSum)')
S, I, R = ss.states('S I R')
infection = S >> I @ 0.3*I
recovery  = I >> R @ ss.days(10)
death = I >> ss.null @ ss.perday(0.01)
exits = recovery + death
try:
    d = ss.Disease(infection, recovery, death, exits, init_prev=0.01, name='sir')
except Exception as e: print('positional FlowSum ERROR:', str(e)[:120])
d = ss.Disease(infection, recovery, death, exits=exits, init_prev=0.01, name='sir')
print('outcomes (keyword):', dict(d.graph.outcomes))

print('\n=== operator edge cases')
S, I, R = ss.states('S I R')
t = I >> R @ ss.days(10) * 2
print('I >> R @ ss.days(10) * 2  -> dur:', t.dur, 'rate:', t.rate, '  (the *2 is silently dropped)')
t = I >> R @ ss.days(10) @ 0.1
print('I >> R @ ss.days(10) @ 0.1 -> dur:', t.dur, 'rate:', t.rate, '  (both set, no error)')
t = I >> R @ 1/ss.days(10)
d = ss.Disease(recovery=t, infection=S >> I @ 0.3*I, init_prev=0.01, name='sir')
sim = ss.Sim(d, unit='day', stop=30, verbose=0).run()
S, I, R = ss.states('S I R')
d2 = ss.Disease(recovery=I >> R @ ss.perday(0.1), infection=S >> I @ 0.3*I, init_prev=0.01, name='sir')
sim2 = ss.Sim(d2, unit='day', stop=30, verbose=0).run()
print('1/ss.days(10) vs ss.perday(0.1): n_R[-1] =', sim.results.sir.n_R[-1], sim2.results.sir.n_R[-1])
S, I, R = ss.states('S I R')
try:
    d3 = ss.Disease(recovery=I >> R @ ss.by(age=[ss.days(5), ss.days(20)]), infection=S >> I @ 0.3*I, init_prev=0.01, name='sir')
    d3.stratify(age=[0, 15])
    print('by of durations after @ -> slot:', d3.recovery.dur, d3.recovery.rate)
    sim3 = ss.Sim(d3, unit='day', stop=30, verbose=0).run()
    print('ran; n_R[-1]', sim3.results.sir.n_R[-1])
except Exception as e: print('ERROR', str(e)[:300])

print('\n--- describe before the sim: units')
S, I, R = ss.states('S I R')
d = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ 10, init_prev=0.01, name='sir')
d.describe()
ss.Sim(d, unit='day', stop=1, verbose=0).init()
d.describe()
