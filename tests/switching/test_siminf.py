"""
Switching demo for the SimInf seat: SIR in 16,000 herds with the movement register, fitted by PMCMC or ABC-SMC.

Passes when: within 3x of SimInf's time, with agreement as today.
Blocked today by: N17.
Scaled down to SimInf's 1,600 herds over the first 100 days (from LFD), a synthetic register of 100 herds, and the one-herd PMCMC example;
and the 16,000-herd time at full size (SimInf 1.9 s to run and 2.1 s to build; v4 52.5 s and 13 s before N26). The step "from R" (N17) is tests/test_r.py.
"""
import sys
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import starsim.fit
sys.path.insert(0, str(sc.thispath().parent)) # For tests/paths.py when run as a script (conftest.py does this under pytest)
import timing

ref_ms_per_run = 0.57 # SimInf's time per particle run in its pmcmc example: switching/siminf.md, "The second week" §3
ref_obs_t = np.arange(5, 101, 5) # SimInf's ?pmcmc example in miniature (s2_data.csv), Poisson observations of I: switching/siminf.md §3
ref_obs_I = [9, 9, 22, 25, 25, 56, 56, 83, 93, 113, 131, 152, 171, 162, 118, 128, 128, 107, 71, 76]
ref_beta, ref_gamma = 0.16, 0.077 # The values the data were simulated with
v4_ms_per_run = 0.37 # Starsim's filter per particle run, in ms, normalized; same date


def make_register(n_herds, n_days, seed=1):
    """ A synthetic register in SimInf's shape: animals entering (births), leaving (deaths), and moving between herds """
    rng = np.random.default_rng(seed)
    n = 6*n_days
    kind = rng.choice(['enter', 'exit', 'move'], n, p=[0.2, 0.2, 0.6])
    src, dst = rng.integers(0, n_herds, n).astype(str), rng.integers(0, n_herds, n).astype(str)
    return pd.DataFrame({'date': np.sort(rng.integers(1, n_days + 1, n)), 'from': np.where(kind == 'enter', None, src),
                         'to': np.where(kind == 'exit', None, dst), 'n': 1, 'states': np.where(kind == 'enter', 'S', None)})


def make_pmcmc():
    """ SimInf's pmcmc example: one herd of 1,000, Poisson observations of I every 5 days """
    beta  = ss.par(ss.perday(0.2), bounds=[0.01, 1])
    gamma = ss.par(ss.perday(0.1), bounds=[0.01, 1])
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, rate=gamma), init_n=dict(I=5), name='sir')
    data = pd.DataFrame(dict(t=ref_obs_t, I=ref_obs_I))
    return ss.Sim(sir, total_pop=1000, solver='gillespie', unit='day', start=1, stop=100, data=data, verbose=0)


def test_siminf_ldata():
    sc.heading('A rate for each herd, as mparse() with ldata')
    n_herds = 100
    b = np.random.default_rng(2).uniform(0.08, 0.24, n_herds)
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(ss.by(herd=b)) * I), recovery=ss.Transition(I, R, rate=ss.perday(0.01)),
                     init_n=dict(I=ss.by(herd=[1] + [0]*(n_herds - 1))), name='sir')
    sir.stratify(herd=np.arange(n_herds).astype(str), kind='space', mobility=ss.geo.Mobility(data=make_register(n_herds, 365)))
    sim = ss.Sim(sir, solver='gillespie', unit='day', start=0, stop=365, total_pop=50*n_herds, verbose=0).run()
    herds = (np.asarray(sim.results.sir.I.strata)[-1] > 0).sum()
    assert herds > 10, f'Expected the register to carry infection to many herds, but {herds} are infected at the end'
    return sim


def test_siminf_filter():
    sc.heading('Timing the particle filter for PMCMC, per particle run')
    pf = ss.fit.ParticleFilter(make_pmcmc().init(), likelihood=dict(I=ss.poisson()))
    pf(10, seed=0) # Compile
    n = 50
    seconds, loglik = timing.timed(lambda: pf(n, seed=1))
    ms_per_run = seconds/n*1e3
    assert np.isfinite(loglik)
    timing.check(ms_per_run, v4_ms_per_run, 'Filter per particle run', unit='ms', other=(ref_ms_per_run, 'SimInf'))
    return ms_per_run


def test_siminf_abc():
    sc.heading('Fitting by ABC-SMC, as SimInf\'s abc()')
    fit = make_pmcmc().fit(method='abc-smc')
    for key, truth in dict(beta=ref_beta, gamma=ref_gamma).items():
        low, high = np.percentile(fit.samples[key], [2.5, 97.5])
        assert low < truth < high, f'The 95% interval of {key}, {low:.3f} to {high:.3f}, misses the truth, {truth}'
    return fit


if __name__ == '__main__':
    sim  = test_siminf_ldata()
    filt = test_siminf_filter()
    fit  = test_siminf_abc()
