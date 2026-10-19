import warnings; warnings.simplefilter('ignore')
import io, contextlib
import starsim as ss
def quiet(f):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): return f()

# 1. bare number in a rate typed per unit
def mk(unit):
    S,E,I,R = ss.states('S E I R')
    R0, gamma = 2.5, 0.2
    seir = ss.Disease(infection=S>>E@R0*gamma*I, latency=E>>I@ss.days(5), recovery=I>>R@ss.days(5), init_prev=0.01, name='seir')
    return ss.Sim(seir, total_pop=10000, unit=unit, stop=ss.days(200), verbose=0)
for u in ['day','week']:
    sim = quiet(lambda: mk(u).run())
    print('1. unit', u, 'peak I', int(sim.results.seir.n_I.values.max()))

# 2. two-line SIR flat?
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10))
sim = quiet(lambda: ss.Sim(sir, stop=100, verbose=0).run())
print('2. two-line SIR max n_I', sim.results.disease.n_I.values.max() if 'disease' in sim.results else list(sim.results.keys()))

# 4. pars assignment
S,I,R = ss.states('S I R'); beta = ss.par(0.05, bounds=[0.01,0.5])
sir = ss.Disease(infection=S>>I@beta*I, recovery=I>>R@ss.days(10), init_prev=0.01, beta=beta)
sir.pars.beta = 0.1
print('4. type after assignment:', type(sir.pars.beta).__name__)

# 5. solver='ode' ignored for code module
sim = quiet(lambda: ss.Sim(diseases=ss.SEIR(), solver='ode', n_agents=500, stop=10, verbose=0).run())
print('5. pars.solver', sim.pars.solver, 'actual', getattr(sim,'solver',None))

# 7. @ ss.days(10) * 2
S,I,R = ss.states('S I R')
t = I >> R @ ss.days(10) * 2
print('7. dur =', t.dur, 'rate =', t.rate)
