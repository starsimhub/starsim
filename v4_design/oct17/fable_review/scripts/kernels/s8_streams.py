"""Does adding an unrelated second disease change the first disease's draws? (spec 11: it must not). Whole-run vs stepped stochastic kernels."""
import numpy as np
import sciris as sc
import starsim as ss

def sir(name):
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=8), init=dict(I=0.01), name=name)

base = dict(unit='day', stop=100, total_pop=10_000, verbose=0, solver='binomial', rand_seed=3)

def run(two, stepped):
    mods = [sir('a')] + ([sir('b')] if two else [])
    intv = [ss.modify('a.pars.beta', multiply=1.0, when=lambda sim: True)] if stepped else []
    sim = ss.Sim(mods, interventions=intv, **base).init()
    assert sim.kernel_only() == (not stepped), sim.kernel_reason()
    return sim.run().results.a.n_I.values

for stepped in [False, True]:
    one, two = run(False, stepped), run(True, stepped)
    print(f'{"stepped" if stepped else "whole-run":9s}: disease a identical with and without disease b: {np.array_equal(one, two)}; a[10:14] alone={one[10:14]} with b={two[10:14]}')

print('=== as agents (declared model, per-distribution streams) ===')
kw = dict(unit='day', stop=100, n_agents=2000, verbose=0, solver='abm', rand_seed=3)
one = ss.Sim([sir('a')], **kw).run().results.a.n_I.values
two = ss.Sim([sir('a'), sir('b')], **kw).run().results.a.n_I.values
print('abm: disease a identical with and without disease b:', np.array_equal(one, two))
