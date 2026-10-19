"""ss.par vs plain numbers vs ss.Pars / SimPars"""
import os
os.environ['MPLBACKEND'] = 'agg'
import numpy as np
import starsim as ss
ss.options.verbose = 0

def sir(beta=0.05):
    return ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', dur=10))

def final_I(sim):
    sim.run()
    return float(sim.results[sim.diseases[0].name].n_I[-1])

# What lives in sir.pars
d = sir()
print('sir.pars type:', type(d.pars).__name__, '| beta entry:', repr(d.pars.beta), type(d.pars.beta).__name__, '| dur entry:', repr(d.pars.dur), type(d.pars.dur).__name__)
print('sir.pars.beta.value =', d.pars.beta.value, '| sir.pars["beta"] is graph par?', d.pars.beta is d.graph.pars.beta)

# Three ways to change beta, do they agree?
base = final_I(ss.Sim(sir(), stop=50))
a = sir(); a.pars.beta = 0.1          # attribute assignment on Pars
b = sir(); b.pars.beta.value = 0.1    # mutate the par
c = sir(); c.set_pars(beta=0.1)
e = sir(); e.pars.update(beta=0.1)
res = dict(base=base,
           attr_assign=final_I(ss.Sim(a, stop=50)),
           value_mutate=final_I(ss.Sim(b, stop=50)),
           set_pars=final_I(ss.Sim(c, stop=50)),
           pars_update=final_I(ss.Sim(e, stop=50)),
           sim_pars=final_I(ss.Sim(sir(), pars=dict(beta=0.1), stop=50)),
           sim_pars_dotted=final_I(ss.Sim(sir(), pars={'sir.beta': 0.1}, stop=50)),
           ctor=final_I(ss.Sim(sir(0.1), stop=50)))
for k, v in res.items():
    print(f'  {k:18s} n_I[-1] = {v:.2f}')
print('  after attr_assign, a.pars.beta is', repr(a.pars.beta), type(a.pars.beta).__name__, '(the ss.par object was replaced by a float; bounds lost?)')

# Starsim-3 style dict pars
try:
    f = final_I(ss.Sim(pars=dict(diseases=sir(), stop=50)))
    print('  Starsim-3 dict pars ok', f)
except Exception as E:
    print('  Starsim-3 dict pars ERROR', type(E).__name__, str(E)[:200])

# Typed pars in sir.pars
d = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.05)), recovery=ss.progression('I -> R', dur=ss.days(10)))
print('typed: pars =', d.pars, '| sir.pars.beta.value =', repr(d.pars.beta.value))
try:
    d.pars.beta = 0.1
    print('  assign bare to typed par: now', repr(d.pars.beta), '-> unit lost?')
except Exception as E:
    print('  assign bare to typed par ERROR', type(E).__name__, str(E)[:200])

# Which pars are fittable: bounds
d = ss.Disease(infection=ss.transmission('S -> I', beta=ss.par(0.05, bounds=[0.01, 0.5])), recovery=ss.progression('I -> R', dur=10))
print('bounds on inline par: beta =', d.pars.beta, '| bounds', d.pars.beta.bounds)
d.pars.beta = 0.2
print('  after sir.pars.beta = 0.2 ->', repr(d.pars.beta), getattr(d.pars.beta, 'bounds', 'NO BOUNDS ATTR'))

# Module with code: pars are the Starsim-3 kind
s3 = ss.SIR(beta=0.05)
print('ss.SIR pars types:', {k: type(v).__name__ for k, v in s3.pars.items()})

# A par used by two modules
beta = ss.par(0.05)
S, I, R = ss.states(3)
d1 = ss.Disease(S >> I @ beta * I, I >> R @ 10)
S2, I2, R2 = ss.states('S2 I2 R2')
d2 = ss.Disease(S2 >> I2 @ beta * I2, I2 >> R2 @ 10, name='d2')
print('shared par: d1.pars.beta is d2.pars.beta?', d1.pars.beta is d2.pars.beta, '| owner module:', beta.module.name)
sim = ss.Sim([d1, d2], pars=dict(beta=0.1), stop=5)
try:
    sim.init()
    print('  sim pars=dict(beta=0.1) with two users ->', d1.pars.beta.value, d2.pars.beta.value, [m.pars.beta.value for m in sim.diseases.values()])
except Exception as E:
    print('  sim pars with shared beta ERROR', type(E).__name__, str(E)[:300])

# Does ss.Sim copy modules? (spec: works on copies)
d = sir()
sim = ss.Sim(d, stop=5).init()
print('ss.Sim copies the module?', sim.diseases[0] is not d, '| par object shared with original?', sim.diseases[0].pars.beta is d.pars.beta)

# Describe what a par prints
p = ss.par(0.05, bounds=[0.01, 0.5], source='Smith 2020', doc='transmission')
print('repr(par):', repr(p), '| str:', str(p), '| after 2*p:', repr(2 * p), type(2 * p).__name__)

# ss.Pars behavior as the user sees it in Starsim 3 modules
print('ss.Pars locked after init? try adding key:')
try:
    sim.diseases[0].pars.newkey = 3
    print('  added newkey without complaint:', sim.diseases[0].pars.newkey)
except Exception as E:
    print('  ', type(E).__name__, str(E)[:150])
