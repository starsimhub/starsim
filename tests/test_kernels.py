"""
Test the kernels: that a run taken in one call matches the same run taken step by step; the backends and threads;
ss.jit; and the exact draws in starsim/solvers/numba/draws.py
"""

# %% Imports and settings
import os
import sys
import subprocess
import pytest
import numpy as np
import sciris as sc
import scipy.special as spec
import scipy.stats as sps
import starsim as ss
from corpus import kernel_models as models, largest_diff

kw = dict(verbose=0)
D = ss.solvers.numba.draws


def run(key, how, **kwargs):
    """ Run a model with a call to the kernels at each step ('step'), or in one call ('all') """
    sim = models[key](**kwargs).init()
    if how == 'all' and key != 'tb':
        assert sim._compiled_reason() is None
    return sim.perf_profile(kind='loop', plot=False).sim if how == 'step' else sim.run() # Recording the time of each step is one of the things that needs the loop


# %% Define the tests

@sc.timer()
def test_ode():
    sc.heading('Testing that the ODE gives the same run in one call to the kernels as step by step')
    for key in ['seir', 'commute', 'malaria', 'tb', 'delay', 'biting']: # Seasonal, stages and a branch; mixing, commuting and aging; counts that are not people; a transfer, so stepped; a conveyor; a weighted count
        diff = largest_diff(run(key, 'all'), run(key, 'step'))
        assert diff < 1e-9, f'{key}: the whole run differs from the stepped one by {diff}' # A value that varies in time is tabulated for a whole run, and worked out at each stage when stepped
    return


@sc.timer()
def test_weighted_count():
    sc.heading('Testing that a weighted count is worked out within each step, as I.n is')
    sim = lambda w=(0.5, 2.0), dt=1, solver='ode': models.biting(w, dt=ss.days(dt), solver=solver).run().results.sir.I.values
    I, fine = sim(), sim(dt=0.25)[::4]
    assert np.abs(I - fine).max()/fine.max() < 1e-5 # Fourth order, not held through each step
    for solver in ['ode', 'binomial']: # Weights of 1 are I.n
        assert np.array_equal(sim((1, 1), solver=solver), sim(None, solver=solver))
    return


@sc.timer()
def test_factored_mixing():
    sc.heading('Testing that a contact matrix within places, with commuting, mixes as the product of the time fractions and the matrix')
    sim = models.commute().init()
    mixing = sim.mixing
    term = mixing.terms[0]
    pre, post = np.kron(mixing.time_frac.toarray(), np.eye(3)), np.kron(mixing.time_frac.toarray(), term.matrix) # The whole matrix between towns by ages, which is not formed
    rng = np.random.default_rng(1)
    infectious, n = rng.random(sim.groups.n_groups), 1 + rng.random(sim.groups.n_groups)
    num, den = [np.bincount(term.cells, weights=v, minlength=9) @ pre for v in [infectious, n]]
    expected = (post @ (num/den))[term.cells]
    assert np.allclose(mixing.term_exposure(term, infectious, n), expected, rtol=1e-12, atol=0)
    assert len(sim.diseases.sir.engine.movers) == 1 and sim.diseases.sir.engine.movers[0].matrix.nnz == 6 # Aging: from each of the first two age groups in each town to the next
    return


@sc.timer()
def test_stages():
    sc.heading('Testing that people pass through as many stages of a dwell time as they should in a step, with another way out: the same as the ODE')
    def cohort(solver):
        E, I, D = ss.states('E I D')
        sim = ss.Sim(ss.Disease(onset=ss.Transition(E, I, dur=ss.normal(7, 1)), death=ss.Transition(E, D, dur=20), default_state='E', name='d'), unit='day', stop=10, total_pop=1_000_000, solver=solver, **kw)
        return sim.run().results.d
    ode, res = cohort('ode'), cohort('binomial')
    assert np.allclose([res.I[6], res.I[8], res.D[10]], [ode.I[6], ode.I[8], ode.D[10]], rtol=0.01)
    return


@sc.timer()
def test_seasonal_steps():
    sc.heading('Testing that the exact solver follows a value that varies within a step')
    def make(solver, dt, seed=0, pop=5000):
        S, I = ss.states('S I')
        sis = ss.Disease(infection=ss.Transition(S, I, rate=ss.seasonal(mean=ss.perday(0.15), amplitude=0.5, peak=0.25) * I), recovery=ss.Transition(I, S, dur=ss.days(10)), init_prev=0.05, name='sis')
        sim = ss.Sim(sis, solver=solver, dt=dt, dur=ss.years(2), total_pop=pop, rand_seed=seed, **kw).run()
        steps = int(round(728/dt.days)) # The first 728 days, eight whole quarters, as the steps from the last time point go past two years
        return sim.results.sis.infection[:steps].sum()/pop, sim.solver
    ode = make('ode', ss.days(1))[0]
    runs = [make('gillespie', ss.days(91), seed) for seed in range(3)] # Steps of a quarter of a year, held at their middle: about 3% too many infections, against 5-6% held at the start
    mean = np.mean([run[0] for run in runs])
    assert str(runs[0][1]) == 'gillespie'
    assert np.isclose(mean, ode, rtol=0.04), f'{mean:n} infections per person, against {ode:n} for the ODE'
    # A step function next to a value that varies smoothly is held for each step, by the ODE stepped as by a whole run
    days = [f'2020-01-{d:02d}' for d in range(1, 32)]
    beta = ss.par(0.5, name='b')*ss.seasonal(mean=0.9, amplitude=0.1)*ss.timeseries(dict(zip(days, np.linspace(1.0, 0.3, len(days))))) # A time series is a step function by default
    def sim():
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, rate=ss.perday(0.25)), init_n=dict(I=100), name='sir')
        return ss.Sim(sir, unit='day', start='2020-01-01', stop='2020-02-15', total_pop=1e6, **kw)
    assert np.allclose(sim().run(until='2020-02-15').results.sir.I, sim().run().results.sir.I, rtol=1e-9)
    return


@sc.timer()
def test_module_beside_kernels():
    sc.heading('Testing that a module with a step of its own does not change a run, and that a parameter it sets is used')
    class Counts(ss.Analyzer):
        def step(self):
            self.last = self.sim.results.sis.I[self.ti] # Results are there for it to read at each step

    class Halves(ss.Intervention):
        def step(self):
            if self.ti == 50:
                self.sim.diseases.sis.pars.beta.value /= 2

    class Replaces(ss.Intervention): # The tables of a whole run no longer hold from then on
        def step(self):
            if self.ti == 50:
                self.sim.diseases.sis.pars.beta = ss.perday(0.05)

    def make(beta=ss.seasonal(mean=ss.perday(0.2), amplitude=0.3), **kwargs):
        return ss.Sim(ss.SIS(beta=beta, dur_inf=ss.days(10), init_prev=0.01), unit='day', stop=200, total_pop=1e5, **kw, **kwargs)

    for solver in ['ode', 'binomial']:
        whole = make(solver=solver).run()
        stepped = make(solver=solver, analyzers=Counts()).run()
        assert np.array_equal(whole.results.sis.I, stepped.results.sis.I), f'{solver}: a module that does nothing changed the run'
        assert stepped.analyzers[0].last == whole.results.sis.I[-1]

    halved = make(beta=ss.perday(0.2), interventions=Halves()).run().results.sis.I
    before = make(beta=ss.perday(0.2)).run().results.sis.I
    assert np.array_equal(halved[:50], before[:50])
    assert halved[-1] < before[-1], 'a parameter set by a module was not used'
    replaced = make(interventions=Replaces()).run()
    assert replaced.results.sis.I[-1] < 0.2*make().run().results.sis.I[-1], 'a value that varied in time was kept after a module replaced it'
    return


@sc.timer()
def test_threads():
    sc.heading('Testing the BLAS threads, even with Starsim imported last, and that a limit on threads in the environment is kept for JAX (see ss.gradients.jax())')
    env = {k:v for k,v in os.environ.items() if k not in ['SCIRIS_NUM_THREADS', 'STARSIM_BLAS_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}
    code = 'import numpy, scipy.linalg, sciris as sc, starsim as ss; print(ss.settings.thread_limit, ss.options.blas_threads, *(getter() for setter,getter in sc.limit_threads.find_libs()))'
    def threads(**extra):
        return subprocess.run([sys.executable, '-c', code], env=env | extra, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1].split()
    for extra,n,limit in [({}, 1, 'None'), (dict(STARSIM_BLAS_THREADS='3', SCIRIS_NUM_THREADS='2'), 3, None), (dict(OMP_NUM_THREADS='2'), None, '2')]: # Starsim's own default of one BLAS thread is not a limit
        out = threads(**extra)
        if n is not None:
            assert set(map(int, out[1:])) == {n}, f'Expected {n} BLAS threads with {extra}, not {out}' # Off Linux, Sciris finds no libraries, so only the option is checked
        if limit is not None:
            assert out[0] == limit
    return


def test_jit():
    sc.heading('Testing a function traced by JAX by ss.jit, and its gradient')
    import jax
    score = lambda x: np.sum(np.log1p(np.exp(-x**2)))
    def longest_run(x):
        """ A loop: the longest run of values above the mean """
        m = x.mean()
        best = cur = 0
        for v in x:
            cur = cur + 1 if v > m else 0
            best = max(best, cur)
        return best
    x = np.random.default_rng(1).normal(size=100)
    with jax.enable_x64(True):
        js = ss.jit(score)
        xj = jax.numpy.asarray(x)
        assert np.isclose(float(js(xj)), score(x), rtol=1e-12)
        assert np.allclose(jax.grad(js)(xj), -2*x*np.exp(-x**2)/(1 + np.exp(-x**2)), rtol=1e-10)
        assert ss.jit(longest_run)(xj) == longest_run(x) # JAX can't trace a loop over values, so it is run as NumPy (with a note)
        with pytest.raises(TypeError):
            jax.grad(ss.jit(lambda x: x[0] if x[0] > 0 else x[1]))(xj) # Under a trace there is no going back to NumPy, so it says why
    return


@pytest.mark.numba # Compiles Numba's draws, so CI runs it only in the jobs for Numba (see pytest.ini)
@sc.timer()
def test_exact_draws():
    sc.heading('Testing the exact draws: exp, log, log1p, expm1, and the inverse normal CDF; the counter-based uniforms; and the inverse CDFs of ss.Dist against SciPy')
    rng = np.random.default_rng(1)
    x = np.concatenate([-rng.uniform(0, 708, 10_000), rng.uniform(0, 709, 1000), -10**rng.uniform(-15, 0, 1000)])
    y = 10**rng.uniform(-300, 300, 10_000)
    z = np.concatenate([rng.uniform(-0.999, 5, 10_000), 10**rng.uniform(-17, -1, 1000)*rng.choice([-1, 1], 1000)])
    u = np.concatenate([rng.random(10_000), 10**rng.uniform(-300, -1, 1000), 1 - 10**rng.uniform(-16, -1, 1000)])
    for f, a, ref in [(D.exp, x, np.exp), (D.log, y, np.log), (D.log1p, z, np.log1p), (D.expm1, x, np.expm1), (D.ndtri, u, spec.ndtri)]:
        out, r = np.array([f(v) for v in a[::50]]), ref(a[::50])
        err = np.max(np.abs(out - r)/np.maximum(np.abs(r), 1e-300 if f is not D.ndtri else 1))
        assert err < 2e-15, f'{f.__name__}: max relative error {err:.1e}'

    # The same hash as ss.Dist's common random numbers
    slots = rng.integers(0, 10**9, 1000)
    assert np.array_equal([D.uniform(D.key(11, 4), s) for s in slots], ss.dists.hash_uniforms(11, 4, slots, dtype=np.float64))
    assert np.array_equal(D.uniforms(D.key(11, 4), 5, np.empty(3)), [D.uniform(D.key(11, 4), i) for i in [5, 6, 7]])

    # The compiled inverse CDFs give what SciPy gives (up to rounding at a step of a discrete CDF, about as likely as a uniform landing on it)
    u = rng.random(20_000)
    n = rng.integers(1, 200, 20_000)
    lam = 10**rng.uniform(-3, 2, 20_000) # Some above a mean of 30, which SciPy takes
    cases = [
        (ss.dists.inverse('expon', u, 3.0), sps.expon.ppf(u, scale=3)),
        (ss.dists.inverse('weibull', u, 1.7, 2.0), sps.weibull_min.ppf(u, 1.7, scale=2)),
        (ss.dists.inverse('binomial', u, n, 0.3, big=n*0.3 > 30, fallback=lambda v, i: sps.binom.ppf(v, n[i], 0.3)), sps.binom.ppf(u, n, 0.3)),
        (ss.dists.inverse('nbinom', u, 2.0, 0.3), sps.nbinom.ppf(u, 2, 0.3)),
        (ss.dists.inverse('poisson', u, lam, big=lam > 30, fallback=lambda v, i: sps.poisson.ppf(v, lam[i])), sps.poisson.ppf(u, lam)),
    ]
    for ours, scipy in cases:
        assert np.allclose(ours, scipy, rtol=1e-12)
    return


if __name__ == '__main__':
    T = sc.timer()
    test_ode()
    test_weighted_count()
    test_factored_mixing()
    test_stages()
    test_seasonal_steps()
    test_module_beside_kernels()
    test_threads()
    test_jit()
    test_exact_draws()
    T.toc()
