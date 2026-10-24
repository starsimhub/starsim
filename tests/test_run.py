"""
Test run, including ss.MultiSim and ss.parallel()
"""

# %% Imports and settings
import pytest
import starsim as ss
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt

n_agents = 1_000
do_plot = False
sc.options(interactive=False) # Assume not running interactively


def make_sim_pars(beta=0.1, **kwargs):
    pars = sc.objdict(
        n_agents = n_agents,
        start = 2000,
        dur = 10,
        demographics = True,
        networks = 'random',
        diseases = sc.objdict(type='sis', beta=beta), # To allow for modification later
    )
    return sc.objdict(pars, **kwargs)


class BySex(ss.Analyzer):
    """ Record a 2D result of infections by sex """
    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('infected', columns=['female', 'male'], label='Infected', kind='stock'))

    def step(self):
        infected = self.sim.diseases.sis.I
        female = self.sim.people.female
        self.results.infected[self.ti] = [np.count_nonzero(infected & female), np.count_nonzero(infected & ~female)]


@sc.timer()
def test_parallel():
    """ Test running two identical sims in parallel """
    sc.heading('Testing parallel...')
    pars = make_sim_pars()

    # Check that two identical sims match
    msim = ss.MultiSim([ss.Sim(pars, label='Sim1'), ss.Sim(pars, label='Sim2')])
    msim.run(shrink=False)
    s1, s2 = msim.sims
    assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True), "Sims don't match and should" # That non-identical sims don't is checked in test_other()

    # Workers started by forkserver, as they are once Numba's or JAX's threads have started here, give the same runs, with a lambda in the sim
    make = lambda: [ss.Sim(make_sim_pars(interventions=lambda sim: None), rand_seed=i) for i in range(2)]
    fork, server = [ss.parallel(make(), par_args=dict(start_method=method)).sims for method in [None, 'forkserver']]
    assert all(np.array_equal(a.results.sis.prevalence, b.results.sis.prevalence) for a,b in zip(fork, server)) # The same either way

    return s1, s2


@sc.timer()
def test_multisim():
    """ Check MultiSim methods """
    sc.heading('Testing MultiSim')
    pars = make_sim_pars()
    msim = ss.MultiSim(ss.Sim(pars))
    msim.init_sims()
    msim.run()

    # The results have a run axis first
    res = msim.results.sis.I
    npts = msim.sims[0].t.npts
    assert len(msim) == 4 and isinstance(res, ss.Result) and res.shape == (4, npts)
    assert np.array_equal(res.select(run=2).values, msim.sims[2].results.sis.I.values) # One run is the result of its sim
    msim.plot() # The median and an interval
    msim.plot(lines=True) # Every run

    # Summaries over the runs
    low, mid, high = res.quantiles([0.1, 0.5, 0.9])
    assert np.all(low <= mid) and np.all(mid <= high)
    assert np.array_equal(res.median().values, np.median(res.values, axis=0))
    res.median().plot()
    assert msim.summarize().loc['sis_I', 'std'] > 0
    msim.show()

    # Export results
    two = msim.results.sis.S.resample(ss.years(2))
    assert np.array_equal(two.values[:, 1], msim.results.sis.S.values[:, 4]) # A stock at the end of each two years from the start
    df = msim.to_df(quantiles=None)
    assert len(df[(df.module == 'sis') & (df.result == 'I')]) == 4*npts # One row per run and time

    # 2D results
    msim2 = ss.MultiSim(ss.Sim(make_sim_pars(analyzers=BySex())), n_runs=3)
    msim2.run(parallel=False)
    res = msim2.results.bysex.infected
    med = res.median()
    assert res.shape == (3, npts, 2) and med.shape == (npts, 2) # The columns are kept
    assert np.array_equal(med.male, np.median(res.values[..., 1], axis=0))
    assert 'infected_male' in set(msim2.to_df().result)
    msim2.plot()

    # Sims that have already run: their results across runs, without running them again
    msim3 = ss.MultiSim(msim.sims)
    assert np.array_equal(msim3.results.sis.I.values, msim.results.sis.I.values)
    with pytest.raises(AttributeError, match='quantiles'): # Starsim 3's msim.reduce() is removed, with what to write instead
        msim3.reduce()

    # A subclass changes the sim of each run, with an option of its own
    class Doubled(ss.MultiSim):
        run_options = ss.MultiSim.run_options + ['beta']
        def make_run(self, sim, values, seed, beta=0.1):
            return super().make_run(sim, dict(values or {}, beta=beta*(1 + seed % 2)), seed)
    doubled = Doubled(ss.Sim(pars), n_runs=2, beta=0.05).run(parallel=False)
    assert sorted(float(s.diseases.sis.pars.beta.value) for s in doubled.sims) == [0.05, 0.1]

    return msim


@sc.timer()
def test_other():
    """ Check other run options """
    sc.heading('Testing other run options')
    # Check parallel
    pars = make_sim_pars()
    pars2 = sc.dcp(pars)
    pars2.diseases.beta *= 2
    s1 = ss.Sim(pars)
    s2 = ss.Sim(pars2)
    ss.parallel(s1, s2, inplace=False)
    assert not s1.initialized
    ss.parallel(s1, s2, inplace=True)
    assert not np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True), "Sims do match and shouldn't"
    return s1,s2


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    s1, s2 = test_parallel()
    msim = test_multisim()
    s3,s4 = test_other()

    T.toc()

    if do_plot:
        plt.show()
