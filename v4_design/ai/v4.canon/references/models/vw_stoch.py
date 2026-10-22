"""
Vynnycky and White (2010), chapter 6: the Reed-Frost chain binomial, and the Greenwood variant

In the Reed-Frost model a case is infectious for exactly one generation, and each susceptible escapes each case independently with probability 1 - p, so
C_{t+1} ~ Binomial(S_t, 1 - (1-p)^C_t). Written in v4 with a step of one generation: an infection hazard of -ln(1-p) per case per step, and a fixed stay of one step.
The reference is a direct numpy simulation; the final-size distributions are compared by a chi-square test.
"""
import numpy as np
import starsim as ss
from scipy import stats

Np, C0, p = 100, 1, 0.02
reps = 400

def build(solver='binomial', seed=0, greenwood=False, **kw):
    S, I, R = ss.states(3)
    if greenwood: # Greenwood: the chance of escape does not depend on the number of cases, only on whether there are any
        rate = ss.perday(-np.log(1-p)) * (I.n > 0)
    else:
        rate = ss.perday(-np.log(1-p)) * I.n
    sir = ss.Disease(infection=S >> I @ rate, recovery=I >> R @ ss.constant(ss.days(1)), init=dict(I=C0), name='rf')
    return ss.Sim(sir, unit='day', dt=1, stop=30, total_pop=Np, solver=solver, rand_seed=seed, verbose=0, **kw)

def reed_frost(reps, rng, greenwood=False):
    out = []
    for r in range(reps):
        S, C, tot = Np - C0, C0, C0
        while C > 0:
            C = rng.binomial(S, p if greenwood else 1 - (1-p)**C)
            S -= C
            tot += C
        out.append(tot)
    return np.array(out)

def final_sizes(solver, reps, **kw):
    out = []
    for seed in range(reps):
        s = build(solver=solver, seed=seed, **kw).run()
        r = s.results.rf
        out.append(r.n_R.values[-1] + r.n_I.values[-1])
    return np.array(out)

def compare(a, b):
    bins = [0, 1.5, 5.5, 20.5, 50.5, 101]
    ha, _ = np.histogram(a, bins); hb, _ = np.histogram(b, bins)
    chi = stats.chi2_contingency(np.array([ha, hb]) + 0.5)
    return ha, hb, chi.pvalue

if __name__ == '__main__':
    ref = reed_frost(20_000, np.random.default_rng(1))
    print(f'Reed-Frost exact: mean final size {ref.mean():.2f}, P(no further case) {np.mean(ref == C0):.3f}, P(major, >20) {np.mean(ref > 20):.3f}')
    s = build().run(); s.describe()
    refg = reed_frost(20_000, np.random.default_rng(2), greenwood=True)
    print(f'Greenwood exact: mean final size {refg.mean():.2f}, P(no further case) {np.mean(refg == C0):.3f}')
    for solver, gw in [('binomial', False), ('abm', False), ('binomial', True)]:
        try:
            fs = final_sizes(solver, reps, greenwood=gw, **(dict(n_agents=Np) if solver == 'abm' else {}))
            if gw: solver, ref = 'greenwood', refg
            ha, hb, pv = compare(fs, ref)
            print(f'{solver:9s} mean {fs.mean():.2f}, P(no further) {np.mean(fs == C0):.3f}, P(major) {np.mean(fs > 20):.3f}; chi-square p = {pv:.3g}; hist {ha} vs exact (scaled) {np.round(hb*reps/len(ref)).astype(int)}')
        except Exception as E:
            print(f'{solver}: {type(E).__name__}: {str(E)[:300]}')
