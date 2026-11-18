"""
Test the kernels: the stochastic solvers whole and stepped, restarts, compiling, ss.jit compiled by Numba, and the exact
draws against their distributions
"""

# %% Imports and settings
import pytest
import numpy as np
import numba as nb
import sciris as sc
import scipy.stats as sps
import starsim as ss
from corpus import kernel_models as models, largest_diff

kw = dict(verbose=0)
D = ss.solvers.numba.draws
N = 200_000 # Draws for each distribution
rng = np.random.default_rng(1)


def run(key, how, **kwargs):
    """ Run a model with a call to the kernels at each step ('step'), or in one call ('all') """
    sim = models[key](**kwargs).init()
    if how == 'all' and key != 'tb':
        assert sim._compiled_reason() is None
    return sim.perf_profile(kind='loop', plot=False).sim if how == 'step' else sim.run()


@sc.timer()
def test_stochastic():
    sc.heading('Testing that the stochastic solvers give the same run in one call to the kernels as step by step, in whole people')
    for solver,pop in dict(binomial=10_000, gillespie=2000).items():
        whole, stepped = [run('commute', how, solver=solver, total_pop=pop) for how in ['all', 'step']]
        assert largest_diff(whole, stepped) == 0, f'{solver}: the whole run differs from the stepped one'
        assert np.all(whole.results.sir.I.values == np.round(whole.results.sir.I.values)) # Whole people

    # Diseases tracked jointly (74 × 51 compartments, with births) are stepped by the compartments that hold anyone: the same events as stepping every flow
    lognormal = [ss.SIR(dur_inf=ss.lognormal(mean=ss.years(6)), p_death=0.01), ss.SIS(dur_inf=ss.lognormal(mean=ss.years(10)))] # Starsim 3's durations, as Erlang stages
    sim = ss.Sim(lognormal, demographics=[ss.birth(rate=ss.peryear(0.05)), ss.death(rate=ss.peryear(0.02))], total_pop=1000, stop=10, solver='gillespie', **kw).init()
    for ti in range(5):
        sim.run_one_step()
    kn, joint = ss.solvers.numba.compartmental, sim.joint_states[0]
    p, places = joint.program, np.arange(joint.program.n_places)
    out = []
    for sparse in [True, False]:
        y, flows = joint.y.copy(), np.zeros((p.n_trans + p.n_movers, joint.y.shape[1]))
        args = (y, p, kn.workspace(y, p), places, 0, len(places), 1.0, flows, np.uint64(0))
        kn.sparse_events(*args, kn.sparse_maps(p, len(y))) if sparse else kn.place_events(*args)
        out.append((y, flows))
    assert np.array_equal(out[0][0], out[1][0]) and np.array_equal(out[0][1], out[1][1])
    return


@sc.timer()
def test_restart():
    sc.heading('Testing restarting a sim (many runs of one are in test_run.py)')

    # A restarted sim gives what a new one does, for another seed or another value
    sim = models.sir(solver='binomial', rand_seed=1).init()
    sim.run()
    sim.restart(rand_seed=5).run()
    assert largest_diff(sim, models.sir(solver='binomial', rand_seed=5).run()) == 0

    # What a restart keeps from the last start is worked out again if a value it reads changes, and only then
    def splits():
        S, E, I, R, D = ss.states(5)
        sir = ss.Disease(S >> E @ ss.par(0.3, name='beta')*I, E >> I @ ss.days(3), ss.Transition(I, {R: ss.par(0.9, name='p_R'), D: None}, dur=ss.days(8), name='end'),
                         init=dict(I=ss.par(0.01, name='I0'), E=0.02), name='sir')
        return ss.Sim(sir, unit='day', stop=40, verbose=0)
    kept = splits().init()
    kept.run()
    engine = kept.diseases.sir.engine
    values = dict()
    for change in [dict(beta=0.4), dict(p_R=0.5), dict(I0=0.05), dict(beta=0.3, p_R=0.9, I0=0.01)]:
        values = values | change # A restart keeps the values of those before it
        kept.restart(pars=change).run()
        assert np.array_equal(kept.results.sir.D, splits().new(pars=values).run().results.sir.D), f'A restart with {values} differs from a new sim'
    kept.restart()
    start = engine.start_y
    assert kept.restart(pars=dict(beta=0.2)).run() and engine.start_y is start, 'A value the start does not read should keep it'
    assert kept.restart(pars=dict(p_R=0.7)).run() and engine.start_y is not start, 'A value the branches read should not keep it'

    # A place where nothing can happen is skipped until something changes it: here, waning that starts on day 30, in a herd where nobody is infected
    wane = 0.05*ss.timeseries({0: 0.0, 29: 0.0, 30: 1.0})
    S, I, R = ss.states('S I R')
    sirs = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3) * I), recovery=ss.Transition(I, R, dur=ss.days(10)), waning=ss.Transition(R, S, rate=wane),
                      init_n=dict(I=ss.by(herd=[5, 0]), R=ss.by(herd=[0, 50])), name='sirs')
    sirs.stratify(herd=ss.levels(['a', 'b'], kind='space'))
    S_b = ss.Sim(sirs, total_pop=200, solver='gillespie', unit='day', stop=60, **kw).run().results.sirs.S.select(herd='b').values
    assert S_b[60] > S_b[29] # Herd b's recovered wane once waning starts
    return


def test_jit_numba():
    sc.heading('Testing a function compiled by Numba by ss.jit, with the same results')
    def longest_run(x):
        """ A loop: the longest run of values above the mean """
        m = x.mean()
        best = cur = 0
        for v in x:
            cur = cur + 1 if v > m else 0
            best = max(best, cur)
        return best
    def draws(n, p, lam, rng):
        """ Array draws from a NumPy Generator, which Numba alone doesn't take """
        return rng.binomial(n, p) + rng.poisson(lam) + rng.binomial(3, 0.5)
    x = np.random.default_rng(1).normal(size=10_000)
    assert ss.jit(longest_run)(x) == longest_run(x)
    score = lambda x: np.sum(np.log1p(np.exp(-x**2)))
    assert np.isclose(ss.jit(score)(x), score(x), rtol=1e-13) # The same up to the order of the sum (NumPy's is pairwise)
    n = np.array([0, 10, 1000, 100_000]*5)
    p = np.linspace(0, 1, 20)
    lam = np.array([0, 0.5, 5, 20, 1000.]*4) # Poisson means of 10 or more are drawn as NumPy does, not as Numba does
    assert np.array_equal(ss.jit(draws)(n, p, lam, np.random.default_rng(2)), draws(n, p, lam, np.random.default_rng(2)))
    assert np.array_equal(ss.jit(draws)(n, p, lam, rng=np.random.default_rng(2)), draws(n, p, lam, rng=np.random.default_rng(2))) # A Generator given by name

    # Arrays broadcast as NumPy's do, and an error if they don't
    def binom(n, p, rng):
        return rng.binomial(n, p)
    jb = ss.jit(binom)
    for n,p in [(np.full((3, 1), 5), np.full((1, 4), 0.3)), (np.array([5]), np.full(3, 0.3))]:
        assert np.array_equal(jb(n, p, np.random.default_rng(3)), binom(n, p, np.random.default_rng(3)))
    with pytest.raises(ValueError):
        jb(np.ones((2, 3), np.int64), np.full((3, 2), 0.5), np.random.default_rng(3))
    return


# %% The exact draws against their distributions

def tv(x, pmf):
    """ The total variation distance between draws and a pmf, and about how big it is from sampling noise alone """
    k = np.arange(x.min(), x.max() + 1)
    emp = np.bincount((x - k[0]).astype(int), minlength=len(k))/len(x)
    p = pmf(k)
    return 0.5*np.abs(emp - p).sum() + 0.5*(1 - p.sum()), np.sqrt(len(k)/len(x))/2


@nb.jit(nopython=True)
def uniforms(k, n):
    """ The first n uniforms of the stream with key k """
    out = np.empty(n)
    for i in range(n):
        out[i] = D.uniform(k, i)
    return out


@nb.jit(nopython=True)
def scalar_binomial(n, p, u, out):
    for i in range(n.size):
        out[i] = D.binomial(n[i], p[i], u[i])
    return out


@nb.jit(nopython=True)
def scalar_poisson(lam, u, out):
    for i in range(lam.size):
        out[i] = D.poisson(lam[i], u[i])
    return out


@sc.timer()
def test_binomial_poisson():
    sc.heading('Testing binomial and Poisson draws: exact, rows the same as scalar, monotone in the uniform')
    w = np.empty((9, N))
    for n, p in [(5, 0.001), (100, 0.003), (100, 0.05), (1000, 0.02), (40, 0.7), (12, 0.708), (20_000, 1e-3), (100_000, 0.2), (10, 1.0), (0, 0.3), (7, 0.0)]:
        nn, pp, u = np.full(N, float(n)), np.full(N, p), rng.random(N)
        x = D.binomials(nn, pp, u, np.empty(N), w)
        d, noise = tv(x, lambda k: sps.binom.pmf(k, n, p))
        assert d < 3*noise + 1e-3, f'Binomial n={n}, p={p}: TV {d:.4f} (noise ~{noise:.4f})'
        assert np.array_equal(x, scalar_binomial(nn, pp, u, np.empty(N))), 'The row and scalar draws differ'
        if n*min(p, 1 - p) <= D.INV_MAX:
            order = np.argsort(u)
            assert np.all(np.diff(x[order]) >= 0), 'Not monotone in the uniform'
    for lam in [0.001, 0.4, 3.0, 9.6, 25.0, 100.0]:
        ll, u = np.full(N, lam), rng.random(N)
        x = D.poissons(ll, u, np.empty(N), w)
        d, noise = tv(x, lambda k: sps.poisson.pmf(k, lam))
        assert d < 3*noise + 1e-3, f'Poisson lam={lam}: TV {d:.4f} (noise ~{noise:.4f})'
        assert np.array_equal(x, scalar_poisson(ll, u, np.empty(N)))

    # Mixed rows, as a model's groups give them
    n = rng.integers(0, 50_000, 1000).astype(float)*(rng.random(1000) > 0.3)
    p = np.where(rng.random(1000) < 0.3, 1.0, 10**rng.uniform(-6, 0, 1000))
    u = rng.random(1000)
    assert np.array_equal(D.binomials(n, p, u, np.empty(1000), w), scalar_binomial(n, p, u, np.empty(1000)))

    # The steps of the rows compile to vector code (a Numba change could silently undo it)
    for step in [D._step_binomial, D._step_poisson]:
        fresh = nb.jit(nopython=True, error_model='numpy')(step.py_func) # Not from the cache, which keeps no assembly
        fresh(*([np.ones(64)]*(6 if step is D._step_poisson else 7)), 1.0, 1.0)
        asm = '\n'.join(fresh.inspect_asm().values())
        assert 'mulpd' in asm, f'{step.__name__} is not vectorized'
    return


@sc.timer()
def test_other_draws():
    sc.heading('Testing the other distributions against their laws')
    for i, (f, pars, law) in enumerate([('expon', (3.0,), sps.expon(scale=3)), ('weibull', (1.7, 2.0), sps.weibull_min(1.7, scale=2)), ('weibull', (0.4, 1.0), sps.weibull_min(0.4))]):
        x = D.many[f](*pars, uniforms(D.key(3, i), N), np.empty(N))
        ks = sps.kstest(x, law.cdf).statistic*np.sqrt(N)
        assert ks < 2.0, f'{f}{pars}: KS statistic x sqrt(N) = {ks:.2f}' # A 0.1% chance by noise alone

    k = D.key(4, 0)
    for n, p in [(0.05, 0.01), (5.0, 0.5), (50.0, 0.4)]:
        x = D.many.nbinom(n, p, uniforms(k, N), np.empty(N))
        d, noise = tv(x, lambda j: sps.nbinom.pmf(j, n, p))
        assert d < 3*noise + 1e-3, f'Negative binomial n={n}, p={p}: TV {d:.4f} (noise ~{noise:.4f})'
    return


def test_caches():
    sc.heading('Testing that the kernels are cached so that they are found again in the next session, and go stale with the draws they inline')
    import os
    compiling = ss.solvers.backends.compiling
    deps = {os.path.basename(k):[os.path.basename(f) for f in v] for k,v in compiling.deps.items()}
    for name in ['passes.py', 'agent_moves.py', 'agent_events.py', 'compartmental.py']:
        assert 'draws.py' in deps.get(name, []), f'{name} does not depend on draws.py: {deps}'
    for name in ['expon', 'weibull', 'binomial', 'poisson', 'nbinom']: # Numba keys its cache by a closure's variables, which change each session, so none is a closure
        func = compiling.numba_kernel('many', name) if ss.solvers.backends.rust.numba else D.many[name]
        assert func.py_func.__closure__ is None, f'draws.many.{name} is a closure, so it is compiled again in every session'
    return


if __name__ == '__main__':
    T = sc.timer()
    test_stochastic()
    test_restart()
    test_jit_numba()
    test_binomial_poisson()
    test_other_draws()
    test_caches()
    T.toc()
