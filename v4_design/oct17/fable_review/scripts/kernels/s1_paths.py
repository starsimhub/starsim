"""Compare a declared SIR run by the Python engine, the Numba kernels (stepped and whole), and JAX."""
import time
import numpy as np
import sciris as sc
import starsim as ss

def make():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=8), init=dict(I=0.01), name='sir')

base = dict(unit='day', stop=100, total_pop=10_000, verbose=0)

def run(how, **kw):
    sim = ss.Sim(make(), **sc.mergedicts(base, kw)).init()
    for mod in sim.declared:
        mod.engine.use_kernels = how != 'python'
    t = time.perf_counter()
    sim.run(profile=(how == 'step'))
    el = time.perf_counter() - t
    return sim, el

print('=== ODE: python / stepped kernels / whole-run kernels / jax ===')
sims = {}
for how, kw in [('python', {}), ('step', {}), ('all', {}), ('jax', dict(integrator='jax')), ('jax2', dict(integrator='jax'))]:
    sim, el = run(how, **kw)
    sims[how] = sim
    print(f'{how:7s} {el*1e3:8.1f} ms  n_I max {sim.results.sir.n_I.max():.6f}  kernel_only={sim.kernel_only()}  notes={[n for n in sim.notes]}')
ref = sims['python'].results.sir.n_I.values
for how in ['step', 'all', 'jax']:
    print(f'  max |n_I - python| for {how}: {np.abs(sims[how].results.sir.n_I.values - ref).max():.2e}')

print()
print('=== Does verbose change the path? ===')
for v in [0, 1]:
    sim = ss.Sim(make(), **sc.mergedicts(base, dict(verbose=v))).init()
    print(f'verbose={v}: kernel_only={sim.kernel_only()}  reason={sim.kernel_reason()!r}')

print()
print('=== binomial solver, same seed, kernels on/off ===')
for seed in [1, 2]:
    out = {}
    for how in ['python', 'all']:
        sim, el = run(how, solver='binomial', rand_seed=seed)
        out[how] = sim.results.sir.n_I.values
        print(f'seed={seed} {how:7s} {el*1e3:8.1f} ms  n_I[10:15]={out[how][10:15]}  final R={sim.results.sir.n_R[-1]:.0f}')
    print(f'  identical: {np.array_equal(out["python"], out["all"])}')

print()
print('=== binomial: does ss.options.crn matter for a compartmental run? ===')
for crn in [True, False]:
    ss.options.crn = crn
    sim, el = run('all', solver='binomial', rand_seed=1)
    print(f'crn={crn}: n_I[10:15]={sim.results.sir.n_I.values[10:15]}')
ss.options.crn = True

print()
print('=== binomial: pairing across scenarios (same seed, modifier from day 50) ===')
a = ss.Sim(make(), solver='binomial', rand_seed=1, **base).run().results.sir.n_I.values
masks = ss.modify('sir.pars.beta', multiply=0.5, start=50)
sim_b = ss.Sim(make(), solver='binomial', rand_seed=1, interventions=masks, **base).init()
print('scenario kernel_only:', sim_b.kernel_only(), sim_b.kernel_reason())
b = sim_b.run().results.sir.n_I.values
first = np.flatnonzero(a != b)
print(f'first step where baseline and scenario differ: {first[0] if len(first) else None} (modifier starts at 50)')

print()
print('=== how many flags ===')
sim = ss.Sim(make(), **base).init()
print('sim.pars.solver', sim.pars.solver, '| sim.solver', sim.solver, '| sim.pars.integrator', sim.pars.integrator, '| sim.integrator', sim.integrator, '| sim.jax', sim.jax, '| engine.use_kernels', sim.diseases.sir.engine.use_kernels, '| pars.substeps', sim.pars.substeps)
