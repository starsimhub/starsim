"""
Test ss.jit: functions compiled by Numba or traced by JAX, a module's methods and a whole coded class compiled with the
same results, what falls back to Python, and the draws compiled code can make (ss.perflib)
"""

# %% Imports and settings
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl

kw = dict(verbose=0)


def longest_run(x):
    """ A loop: the longest run of values above the mean """
    m = x.mean()
    best = cur = 0
    for v in x:
        cur = cur + 1 if v > m else 0
        best = max(best, cur)
    return best


def score(x):
    """ Elementwise, then summed """
    return np.sum(np.log1p(np.exp(-x**2)))


def draws(n, p, lam, rng):
    """ Array draws from a NumPy Generator, which Numba alone doesn't take """
    return rng.binomial(n, p) + rng.poisson(lam) + rng.binomial(3, 0.5)


class Severe(ssl.AgentSIR):
    """ AgentSIR plus severe disease, written as code, with its methods compiled by ss.jit (or run as written) """
    def __init__(self, compiled=True, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(p_severe=ss.bernoulli(0.3), dur_sev=ss.lognorm_ex(mean=ss.years(0.5)), sev_factor=3.0)
        self.define_states(ss.BoolState('severe', label='Severe'), ss.FloatArr('ti_severe', label='Time of severe'))
        self.compiled = compiled

    def method(self, name):
        meth = getattr(Severe, name)
        return meth.__get__(self) if self.compiled else meth.py_func.__get__(self)

    def set_progression(self, uids):
        super().set_progression(uids)
        self.method('schedule')(uids)

    def step_state(self):
        self.method('progress')()
        deaths = self.method('progress_loop')()
        self.sim.people.request_death(deaths, cause=self)

    @ss.jit
    def schedule(self, uids):
        p = self.pars
        sev = p.p_severe.filter(uids)
        self.ti_severe[sev] = self.ti_infected[sev] + p.dur_sev.rvs(sev)

    @ss.jit
    def progress(self):
        recovered = (self.infected & (self.ti_recovered <= self.ti)).uids
        self.infected[recovered] = False
        self.severe[recovered] = False
        self.recovered[recovered] = True
        self.rel_trans[:] = 1.0
        self.rel_trans[self.severe] = self.pars['sev_factor']

    @ss.jit
    def progress_loop(self):
        for uid in self.infected.uids:
            if self.ti_severe[uid] <= self.ti:
                self.severe[uid] = True
                self.rel_trans[uid] = self.pars.sev_factor
        return (self.ti_dead <= self.ti).uids


class Coded(ssl.AgentSEIR):
    """ A disease written as code, as modelers write it: calls to super() and other methods, a rate, UIDs combined, and a probability for each agent """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(waning=ss.peryear(0.5), p_test=ss.bernoulli(p=0))
        self.define_states(ss.BoolState('tested', label='Tested'), ss.FloatArr('level', default=0.0))
        return

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('mean_level', dtype=float, scale=False))
        return

    def step_state(self):
        super().step_state() # AgentSEIR's, itself compiled in parts (AgentSIR's request_death() runs as Python)
        decay = self.pars.waning.to_prob() # A rate as a number, worked out before the compiled part
        self.level[self.recovered] *= 1 - decay
        cand = (self.infectious.uids | self.exposed.uids) - self.tested.uids # Union, then remove, as ss.uids does them
        self.pars.p_test.set(p=self.level[cand]*0.5) # A probability for each agent, set in Python after the compiled part
        self.tested[self.pars.p_test.filter(cand)] = True
        return

    def set_prognoses(self, uids, sources=None):
        super().set_prognoses(uids, sources) # AgentSIR's: Starsim's own super().set_prognoses() runs as Python, then its own calls are compiled in
        self.level[uids] = 1.0
        return

    def update_results(self):
        super().update_results()
        self.results['mean_level'][self.ti] = self.level.mean() # Summed as NumPy sums, so the same to the last bit
        return


# %% Define the tests


def test_jax():
    sc.heading('Testing a function traced by JAX, and its gradient')
    import jax
    jnp = jax.numpy
    x = np.random.default_rng(1).normal(size=100)
    with jax.enable_x64(True):
        js = ss.jit(score)
        xj = jnp.asarray(x)
        assert np.isclose(float(js(xj)), score(x), rtol=1e-12)
        grad = jax.grad(js)(xj)
        expected = -2*x*np.exp(-x**2)/(1 + np.exp(-x**2))
        assert np.allclose(grad, expected, rtol=1e-10)
        jl = ss.jit(longest_run)
        assert jl(xj) == longest_run(x) # JAX can't trace a loop over values, so it is run as NumPy (with a note)
        with pytest.raises(TypeError):
            jax.grad(ss.jit(lambda x: x[0] if x[0] > 0 else x[1]))(xj) # Under a trace there is no going back to NumPy, so it says why
    return


def test_method():
    sc.heading('Testing a module\'s methods compiled, with the same results as when run as Python')
    def run(compiled):
        d = Severe(compiled=compiled, beta=ss.peryear(0.8), dur_inf=ss.lognorm_ex(mean=ss.years(1)), p_death=ss.bernoulli(0.05))
        return ss.Sim(n_agents=5000, diseases=d, networks='random', demographics=True, stop=2030, rand_seed=3, **kw).run()
    s0, s1 = run(False), run(True)
    assert s1.people.n_uids > s1.people.n_alive # Some have died, so the active agents are not everyone
    for key in ['infected', 'recovered', 'severe', 'new_infections']:
        assert np.array_equal(s0.results.severe[key], s1.results.severe[key]), f'{key} differs when compiled'
    assert np.array_equal(s0.diseases.severe.rel_trans.values, s1.diseases.severe.rel_trans.values)
    for name in ['schedule', 'progress', 'progress_loop']:
        meth = getattr(Severe, name)
        assert meth.failed is None and meth.plan is not None, f'{name}() was not compiled: {meth.failed}'
    return


def test_class():
    sc.heading('Testing a coded disease class compiled unchanged, with the same results')
    pars = dict(beta=ss.peryear(0.8), dur_inf=ss.lognorm_ex(mean=ss.years(1)), p_death=ss.bernoulli(0.05))
    def run(cls):
        return ss.Sim(n_agents=3000, diseases=cls(**pars), networks='random', demographics=True, stop=2025, rand_seed=2, **kw).run()
    Compiled = ss.jit(type('Coded', (Coded,), {})) # A subclass, so Coded itself stays as written
    s0, s1 = run(Coded), run(Compiled)
    r0, r1 = s0.diseases[0].results, s1.diseases[0].results
    for key in ['exposed', 'infectious', 'recovered', 'tested', 'new_infections', 'mean_level']:
        assert np.array_equal(r0[key], r1[key]), f'{key} differs when compiled'
    assert r1.tested[-1] > 0 and s1.people.n_uids > s1.people.n_alive
    from starsim.perflib.jit import Segment # The module; ss.perflib.jit is the function
    parts = {name:getattr(Compiled, name).plan.parts for name in ['step_state', 'set_prognoses', 'step_die', 'update_results']}
    for name, p in parts.items():
        assert any(isinstance(x, Segment) and not x.failed for x in p), f'{name}() has no compiled part'
    assert Compiled.step_die.plan.whole is not None # With self.clear_infection() compiled in
    assert Compiled.set_prognoses.plan.whole is None # Starsim's own super().set_prognoses() runs as Python, the rest is compiled
    return


def test_fast_draws():
    sc.heading('Testing the draws compiled code can make: the same from Python and compiled, and the right distribution')
    k = ss.perflib.key(1, 2)
    m = 200_000
    u = ss.perflib.uniform(k, np.arange(m))
    assert u.min() > 0 and u.max() < 1 and abs(u.mean() - 0.5) < 0.005
    compiled = ss.jit(lambda n, p, lam, k: (ss.perflib.binomial(n, p, ss.perflib.uniform(k, np.arange(n.size))), ss.perflib.poisson(lam, ss.perflib.uniform(k, np.arange(n.size)))))
    for n, p, lam in [(20, 0.3, 0.5), (10_000, 0.001, 8.0), (5000, 0.9, 400.0)]: # Inversion from 0, and from the mode
        ns, lams = np.full(m, n), np.full(m, lam)
        b, q = ss.perflib.binomial(ns, p, u), ss.perflib.poisson(lams, u)
        b2, q2 = compiled(ns, p, lams, k)
        assert np.array_equal(b, b2) and np.array_equal(q, q2)
        for x, mean, var in [(b, n*p, n*p*(1 - p)), (q, lam, lam)]:
            assert abs(x.mean() - mean) < 5*np.sqrt(var/m) and abs(x.var()/var - 1) < 0.03, f'{x.mean()}, {x.var()} against {mean}, {var}'
    return


if __name__ == '__main__':
    T = sc.timer()
    test_jax()
    test_method()
    test_class()
    test_fast_draws()
    T.toc()
