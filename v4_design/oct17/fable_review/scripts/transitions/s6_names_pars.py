"""Names from variables, and parameters."""
import starsim as ss, numpy as np, sciris as sc

print('=== inline transitions: fallback names')
S, I, R = ss.states(3)
d = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init_prev=0.01)
print(list(d.transitions.keys()), '| pars:', dict(d.pars), '| guessed:', [(k, t.guessed) for k,t in d.transitions.items()])
print('results names:', [k for k in ss.Sim(d, stop=5, verbose=0).run().results[d.name].keys()])

print('\n=== states in a list (no variable per state)')
try:
    sts = ss.states(3)
    d = ss.Disease(sts[0] >> sts[1] @ 0.3*sts[1], sts[1] >> sts[2] @ ss.days(10))
except Exception as e: print('ERROR', str(e)[:300])

print('\n=== one state bound to two variables')
S, I, R = ss.states(3)
Inf = I
try:
    d = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10))
    print('states:', list(d.graph.states.keys()))
except Exception as e: print('ERROR', str(e)[:300])

print('\n=== a transition bound to two variables')
S, I, R = ss.states('S I R')
infection = S >> I @ 0.3*I
inf2 = infection
recovery = I >> R @ ss.days(10)
d = ss.Disease(infection, recovery)
print('transitions:', list(d.transitions.keys()))

print('\n=== built inside a function (local frame), with states from module scope')
def make():
    S, I, R = ss.states(3)
    infection = S >> I @ 0.3*I
    recovery = I >> R @ ss.days(10)
    return ss.Disease(infection, recovery, init_prev=0.01)
d = make(); print('states:', list(d.graph.states.keys()), 'trans:', list(d.transitions.keys()))

print('\n=== built by a helper that takes the transitions as arguments (frame is the helper)')
def helper(*trs, **kw):
    return ss.Disease(*trs, **kw)
S, I, R = ss.states(3)
infection = S >> I @ 0.3*I
recovery = I >> R @ ss.days(10)
try:
    d = helper(infection, recovery)
    print('states:', list(d.graph.states.keys()), 'trans:', list(d.transitions.keys()))
except Exception as e: print('ERROR', str(e)[:300])

print('\n=== exec / REPL-like: globals only')
ns = {}
exec("import starsim as ss\nS, I, R = ss.states(3)\ninfection = S >> I @ 0.3*I\nrecovery = I >> R @ ss.days(10)\nd = ss.Disease(infection, recovery)", ns)
print('states:', list(ns['d'].graph.states.keys()))

print('\n=== parameter naming')
S, I, R = ss.states('S I R')
beta = ss.par(0.3, bounds=[0.1, 1])
d = ss.Disease(infection=S >> I @ beta * I, recovery=I >> R @ ss.days(10), death=I >> ss.null @ ss.perday(0.001), init_prev=0.01)
print('pars:', dict(d.pars))
print('d.infection.beta:', getattr(d.infection, 'beta', 'MISSING'), '| d.recovery.dur:', d.recovery.dur, '| d.death.rate:', d.death.rate)
print('d.pars.beta is beta:', d.pars.beta is beta)
S, I, R = ss.states('S I R')
d2 = ss.Disease(infection=S >> I @ 0.3 * I, recovery=I >> R @ ss.days(10), relapse=R >> I @ ss.days(100), init_prev=0.01)
print('two anonymous durs:', dict(d2.pars))
S, I, R = ss.states('S I R')
d3 = ss.Disease(infection=S >> I @ ss.par(0.3, bounds=[0.1, 1]) * I, recovery=I >> R @ ss.par(ss.days(10), bounds=[3, 30]), init_prev=0.01)
print('ss.par written inline in the slot:', dict(d3.pars), [(k, p.bounds) for k,p in d3.pars.items()])
print('string form:', dict(ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10)).pars))
S, I, R = ss.states('S I R')
gamma = 0.1
d4 = ss.Disease(infection=S >> I @ 0.3 * I, recovery=I >> R @ gamma, init_prev=0.01)
print('plain python float gamma in a slot:', dict(d4.pars), '(gamma is a bare float so not found by name)')
S, I, R = ss.states('S I R')
R0, gamma = ss.par(3.0), ss.par(0.1)
d5 = ss.Disease(infection=S >> I @ R0 * gamma * I, recovery=I >> R @ gamma, init_prev=0.01)
print('R0*gamma*I:', dict(d5.pars))
d5.describe()
print('json pars:', d5.to_json()['pars'])

print('\n=== set_pars / E602')
try: d5.set_pars(gama=0.2)
except Exception as e: print('ERROR', e)
d5.set_pars(gamma=0.2); print('after set_pars gamma:', d5.pars.gamma, 'raw', d5.pars.gamma.raw, 'recovery.rate.raw', d5.recovery.rate.raw)
d5.pars.gamma = 0.3; print('after pars.gamma = 0.3:', d5.pars.gamma, type(d5.pars.gamma), 'recovery.rate.raw', d5.recovery.rate.raw)

print('\n=== ss.par with bounds and no value')
S, I, R = ss.states('S I R')
b = ss.par(bounds=[0.1, 1.0])
d6 = ss.Disease(infection=S >> I @ b * I, recovery=I >> R @ ss.days(10), init_prev=0.01)
print('pars:', dict(d6.pars))
d6.describe()

print('\n=== Sim(pars=dict)')
sim = ss.Sim(sc.dcp(d6), pars=dict(b=0.5), unit='day', stop=5, verbose=0); sim.init()
print('b after Sim(pars=):', sim.diseases[d6.name].pars.b)
try:
    ss.Sim(sc.dcp(d6), pars=dict(bb=0.5), unit='day', stop=5, verbose=0).init()
except Exception as e: print('ERROR', str(e)[:200])

print('\n=== keyword parameter that is a plain number: ss.Disease(beta=0.3, ...)')
d7 = ss.Disease(beta=0.3, infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10))
print('pars:', dict(d7.pars))
