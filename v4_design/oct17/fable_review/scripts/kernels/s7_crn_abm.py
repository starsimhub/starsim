"""Is CRN meaningful for a declared model run as agents? Baseline vs scenario that diverges at day 50."""
import numpy as np
import sciris as sc
import starsim as ss

def make():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=8), init=dict(I=0.01), name='sir')

kw = dict(unit='day', stop=100, n_agents=2000, verbose=0, solver='abm', rand_seed=1)
for crn in [True, False]:
    ss.options.crn = crn
    a = ss.Sim(make(), **kw).run()
    m = ss.modify('sir.pars.beta', multiply=0.5, start=50)
    b = ss.Sim(make(), interventions=m, **kw).run()
    ai, bi = a.results.sir.n_I.values, b.results.sir.n_I.values
    first = np.flatnonzero(ai != bi)
    # Who is infected by day 49 in each: identical agents?
    same_people = np.array_equal(a.diseases.sir.I.uids, a.diseases.sir.I.uids)
    print(f'crn={crn}: first differing step {first[0] if len(first) else None}; n_I[45:55] a={ai[45:55]} b={bi[45:55]}')
    # Variance of the difference over seeds
    diffs = []
    for seed in range(5):
        kw2 = dict(kw, rand_seed=seed)
        a = ss.Sim(make(), **kw2).run().results.sir.cum_infection.values[-1]
        b = ss.Sim(make(), interventions=ss.modify('sir.pars.beta', multiply=0.5, start=50), **kw2).run().results.sir.cum_infection.values[-1]
        diffs.append(a - b)
    print(f'   effect (cum infections averted) over 5 seeds: mean {np.mean(diffs):.1f}, sd {np.std(diffs):.1f}')
ss.options.crn = True

print('=== same thing with solver=binomial: run-level pairing ===')
for seed_shift in [0]:
    diffs = []
    for seed in range(5):
        kw2 = dict(unit='day', stop=100, total_pop=2000, verbose=0, solver='binomial', rand_seed=seed)
        a = ss.Sim(make(), **kw2).run().results.sir.cum_infection.values[-1]
        b = ss.Sim(make(), interventions=ss.modify('sir.pars.beta', multiply=0.5, start=50), **kw2).run().results.sir.cum_infection.values[-1]
        diffs.append(a - b)
    print(f'   binomial, paired seeds: mean {np.mean(diffs):.1f}, sd {np.std(diffs):.1f}')
    diffs = []
    for seed in range(5):
        a = ss.Sim(make(), **dict(unit='day', stop=100, total_pop=2000, verbose=0, solver='binomial', rand_seed=seed)).run().results.sir.cum_infection.values[-1]
        b = ss.Sim(make(), interventions=ss.modify('sir.pars.beta', multiply=0.5, start=50), **dict(unit='day', stop=100, total_pop=2000, verbose=0, solver='binomial', rand_seed=seed + 100)).run().results.sir.cum_infection.values[-1]
        diffs.append(a - b)
    print(f'   binomial, unpaired seeds: mean {np.mean(diffs):.1f}, sd {np.std(diffs):.1f}')
