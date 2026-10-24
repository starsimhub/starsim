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


# %% Define the tests

def test_function():
    sc.heading('Testing a function compiled by Numba, with the same results')
    x = np.random.default_rng(1).normal(size=10_000)
    assert ss.jit(longest_run)(x) == longest_run(x)
    assert np.isclose(ss.jit(score)(x), score(x), rtol=1e-13) # The same up to the order of the sum (NumPy's is pairwise)
    n = np.array([0, 10, 1000, 100_000]*5)
    p = np.linspace(0, 1, 20)
    lam = np.array([0, 0.5, 5, 20, 1000.]*4) # Poisson means of 10 or more are drawn as NumPy does, not as Numba does
    out = ss.jit(draws)(n, p, lam, np.random.default_rng(2))
    assert np.array_equal(out, draws(n, p, lam, np.random.default_rng(2)))

    @ss.jit
    def uncompilable(x):
        return sc.findinds(x > 0) # Numba can't call Sciris
    assert np.array_equal(uncompilable(x), sc.findinds(x > 0)) # Runs as Python, with the same result
    assert uncompilable.failed and uncompilable.said == {None}
    return


if __name__ == '__main__':
    T = sc.timer()
    test_function()
    T.toc()
