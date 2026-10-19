"""Reproducibility of stochastic runs that are stepped (not whole-run), integrator='scipy', and whether describe mentions CRN."""
import numpy as np
import sciris as sc
import starsim as ss

def make():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=8), init=dict(I=0.01), name='sir')

base = dict(unit='day', stop=100, total_pop=10_000, verbose=0, solver='binomial', rand_seed=3)

print('=== stepped stochastic run (forced by a when= modifier): same seed twice in one process ===')
def stepped():
    m = ss.modify('sir.pars.beta', multiply=0.5, when=lambda sim: sim.ti >= 20)
    sim = ss.Sim(make(), interventions=m, **base).init()
    assert not sim.kernel_only(), sim.kernel_reason()
    return sim.run().results.sir.n_I.values
a, b = stepped(), stepped()
print('identical:', np.array_equal(a, b), a[20:25], b[20:25])

print('=== whole-run stochastic: same seed twice ===')
a = ss.Sim(make(), **base).run().results.sir.n_I.values
b = ss.Sim(make(), **base).run().results.sir.n_I.values
print('identical:', np.array_equal(a, b))

print('=== stepped stochastic via verbose=1 vs whole-run, same seed ===')
c = ss.Sim(make(), **sc.mergedicts(base, dict(verbose=0.5))).run().results.sir.n_I.values  # verbose>=1 forces the loop; 0.5 does not?
sim = ss.Sim(make(), **sc.mergedicts(base, dict(verbose=1))).init()
print('verbose=1 kernel_only:', sim.kernel_only())
d = sim.run().results.sir.n_I.values
print('whole vs verbose=1 identical:', np.array_equal(a, d), '| whole vs verbose=0.5 identical:', np.array_equal(a, c))

print('=== Does a stepped stochastic run with two sims interleave the global Numba RNG? (run A alone vs A then B then A) ===')
e = stepped()
print('stepped A repeated after other runs identical to first:', np.array_equal(a_s := stepped(), e))

print('=== integrator="scipy" (as the spec spells it) ===')
try:
    sim = ss.Sim(make(), unit='day', stop=100, total_pop=10_000, verbose=0, integrator='scipy').run()
    print('ok', sim.results.sir.n_I.max())
except Exception as E:
    print('error:', type(E).__name__, str(E)[:300])
print('=== integrator="RK45" ===')
sim = ss.Sim(make(), unit='day', stop=100, total_pop=10_000, verbose=0, integrator='RK45').init()
print('kernel_only:', sim.kernel_only(), '|', sim.kernel_reason())
sim.run(); print('ok', sim.results.sir.n_I.max())

print('=== integrator="jax" on a sim it cannot run (when= modifier) ===')
m = ss.modify('sir.pars.beta', multiply=0.5, when=lambda sim: sim.ti >= 20)
sim = ss.Sim(make(), interventions=m, unit='day', stop=100, total_pop=10_000, verbose=0, integrator='jax').run()
print('notes:', sim.notes)

print('=== solver given at run() vs Sim(); integrator at run()? ===')
sim = ss.Sim(make(), unit='day', stop=100, total_pop=10_000, verbose=0)
try:
    sim.run(solver='binomial'); print('run(solver=) ok, solver =', sim.solver)
except Exception as E:
    print('run(solver=) error:', E)
sim = ss.Sim(make(), unit='day', stop=100, total_pop=10_000, verbose=0)
try:
    sim.run(integrator='euler'); print('run(integrator=) ok')
except Exception as E:
    print('run(integrator=) error:', type(E).__name__, str(E)[:200])
