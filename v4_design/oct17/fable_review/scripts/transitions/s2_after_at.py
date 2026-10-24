"""What may follow @, and arithmetic on transitions."""
import starsim as ss, numpy as np, sciris as sc

def show(label, fn):
    print(f'\n--- {label}')
    try:
        tr = fn()
        print('  ->', tr, '| rate:', tr.rate, '| dur:', tr.dur)
        return tr
    except Exception as e:
        print('  ERROR', type(e).__name__, str(e)[:200])

S, I, R = ss.states('S I R')
show('@ 10 (bare number: a rate of 10 per step!)', lambda: I >> R @ 10)
show('@ ss.days(10)', lambda: I >> R @ ss.days(10))
show('@ ss.perday(0.1)', lambda: I >> R @ ss.perday(0.1))
show('@ ss.probperday(0.1)', lambda: I >> R @ ss.probperday(0.1))
show('@ ss.par(ss.days(10))', lambda: I >> R @ ss.par(ss.days(10)))
show('@ ss.par(10)  (par of bare number: rate)', lambda: I >> R @ ss.par(10))
show('@ ss.lognorm_ex(mean=10, std=3)  (any Dist -> dur, even unitless)', lambda: I >> R @ ss.lognorm_ex(mean=10, std=3))
show('@ ss.bernoulli(0.5) (a Dist, so... a dwell time?)', lambda: I >> R @ ss.bernoulli(0.5))
show('@ ss.erlang(k=3, mean=ss.days(10))', lambda: I >> R @ ss.erlang(k=3, mean=ss.days(10)))
show('@ lambda sim: 0.1 (function -> rate)', lambda: I >> R @ (lambda sim: 0.1))
show('@ ss.by(age=[1,2,3])  (by of bare numbers -> rate)', lambda: I >> R @ ss.by(age=[1, 2, 3]))
show('@ ss.by(age=[ss.days(1), ss.days(2)]) (by of durations -> ?)', lambda: I >> R @ ss.by(age=[ss.days(1), ss.days(2)]))
show('@ ss.timeseries({0: ss.days(5), 10: ss.days(10)}) (timeseries of durations -> dur)', lambda: I >> R @ ss.timeseries({0: ss.days(5), 10: ss.days(10)}))
show('@ ss.timeseries({0: 5, 10: 10}) (timeseries of bare -> rate)', lambda: I >> R @ ss.timeseries({0: 5, 10: 10}))
show('@ dict(mf=0.1) (beta dict after @)', lambda: S >> I @ dict(mf=0.1) * I)
show('@ 0.3 * I  (with I a bare state)', lambda: S >> I @ 0.3 * I)
show('@ ss.days(10) * 2 (arithmetic on a duration)', lambda: I >> R @ ss.days(10) * 2)
show('@ 1 / ss.days(10)  (one over a duration: is it a rate?)', lambda: I >> R @ 1 / ss.days(10))
show('(I >> R) @ ss.days(10)  (parenthesised)', lambda: (I >> R) @ ss.days(10))
show('I >> R @ ss.days(10) @ 0.1 (two @)', lambda: I >> R @ ss.days(10) @ 0.1)
show('S >> I @ 0.3 * I @ 2', lambda: S >> I @ 0.3 * I @ 2)
show('S >> I @ 0.3 * I * 2', lambda: S >> I @ 0.3 * I * 2)
show('S >> I @ 0.3 * I / (S + I + R)  (explicit /N: nonlinear? no: /N.n fine, bare N is nonlinear)', lambda: S >> I @ 0.3 * I / (S + I + R))
show('S >> I @ 0.3 * I / (S + I + R).n', lambda: S >> I @ 0.3 * I / (S + I + R).n)
show('I ** 2 in rate', lambda: S >> I @ 0.3 * I**2)

print('\n=== spec says: "infection * 0.5 is the same transition at half the rate"')
inf = S >> I @ 0.3 * I
half = inf * 0.5
print('inf*0.5 ->', half, 'rate:', half.rate, 'flow_of:', half.flow_of, 'name:', half.name)
half2 = 0.5 * inf
print('0.5*inf ->', half2, 'rate:', half2.rate, 'flow_of:', half2.flow_of)
unfinished = (I @ 0.3) * 2   # transition with no src yet
print('(I @ 0.3)*2 ->', unfinished, 'rate:', unfinished.rate, 'flow_of:', unfinished.flow_of)

# Use infection*0.5 as a transition in a disease: what happens?
S, I, R = ss.states('S I R')
inf = S >> I @ 0.3 * I
rec = I >> R @ ss.days(10)
try:
    d = ss.Disease(infection=inf*0.5, recovery=rec, init_prev=0.01)
    d.describe()
    print('--> built; infection rate =', d.infection.rate, '; outcomes:', dict(d.graph.outcomes))
except Exception as e:
    print('ERROR', e)

# Model with a bare number after @ run in days
S, I, R = ss.states('S I R')
d = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ 10, init_prev=0.01)
sim = ss.Sim(d, unit='day', stop=30, verbose=0).run()
print('\nI >> R @ 10 run for 30 days: peak I =', sim.results[d.name].n_I.max(), '(rate 10/day: infection never takes off)')
