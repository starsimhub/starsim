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
        self.define_results(ss.Result('n_infected', columns=['female', 'male'], label='Infected'))

    def step(self):
        infected = self.sim.diseases.sis.I
        female = self.sim.people.female
        self.results.n_infected[self.ti] = [np.count_nonzero(infected & female), np.count_nonzero(infected & ~female)]


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
    res = msim.results.sis.n_I
    npts = msim.sims[0].t.npts
    assert len(msim) == 4 and isinstance(res, ss.Result) and res.shape == (4, npts)
    assert np.array_equal(res.select(run=2).values, msim.sims[2].results.sis.n_I.values) # One run is the result of its sim
    msim.plot() # The median and an interval
    msim.plot(lines=True) # Every run

    # Summaries over the runs
    low, mid, high = res.quantiles([0.1, 0.5, 0.9])
    assert np.all(low <= mid) and np.all(mid <= high)
    assert np.array_equal(res.median().values, np.median(res.values, axis=0))
    res.median().plot()
    assert msim.summarize().sis_n_I.std > 0
    msim.show()

    # Export results
    two = msim.results.sis.n_S.resample(ss.years(2))
    assert np.array_equal(two.values[:, 1], msim.results.sis.n_S.values[:, 4]) # A stock at the end of each two years from the start
    df = msim.to_df(quantiles=None)
    assert len(df[(df.module == 'sis') & (df.result == 'n_I')]) == 4*npts # One row per run and time

    # 2D results
    msim2 = ss.MultiSim(ss.Sim(make_sim_pars(analyzers=BySex())), n_runs=3)
    msim2.run(parallel=False)
    res = msim2.results.bysex.n_infected
    med = res.median()
    assert res.shape == (3, npts, 2) and med.shape == (npts, 2) # The columns are kept
    assert np.array_equal(med.male, np.median(res.values[..., 1], axis=0))
    assert 'n_infected_male' in set(msim2.to_df().result)
    msim2.plot()

    # Sims that have already run: their results across runs, without running them again
    msim3 = ss.MultiSim(msim.sims)
    assert np.array_equal(msim3.results.sis.n_I.values, msim.results.sis.n_I.values)
    with pytest.raises(AttributeError): # Starsim 3's msim.reduce() is removed, with what to write instead
        msim3.reduce()

    return msim


@sc.timer()
def test_multisim_construction(do_plot=False):
    """ Test other MultiSim construction methods """
    sc.heading('Testing MultiSim construction')
    
    # Make sims
    def make_sims(n=2):
        sims = sc.autolist()
        for beta in 0.1*np.random.rand(n):
            pars = make_sim_pars(beta=beta, rand_seed=np.random.randint(999))
            sims += ss.Sim(pars=pars, label=f'{beta=}')
        return sims
    
    # Construction options
    m1 = ss.MultiSim(make_sims(), label='list') # As list
    m2 = ss.MultiSim(tuple(make_sims()), label='tuple') # As tuple
    m3 = ss.MultiSim(m1.copy(), label='msim') # As MultiSim
    m4 = ss.MultiSim([m1.copy(), m2.copy()], label='multi-msim') # As list of MultiSims
    
    # Check that run works
    for msim in [m1, m2, m3, m4]:
        msim.run(parallel=msim is m4) # Only one in parallel, to save starting workers
        if do_plot:
            msim.plot()
    
    return m4


@sc.timer()
def test_combine_merge():
    """ Test MultiSim combine(), compare(), merge() and split() """
    sc.heading('Testing MultiSim combine, compare, merge, and split')
    msim = ss.MultiSim(ss.Sim(make_sim_pars(analyzers=BySex())), n_runs=3)
    msim.run(parallel=False)
    sims = msim.sims

    # Combine
    sim = msim.combine()
    res = sim.results
    assert sim.pars.n_agents == 3*n_agents and sim.pars.pop_scale == 1 # Population sizes are summed
    assert res.sis.n_I[-1] == sum(s.results.sis.n_I[-1] for s in sims) # Counts are summed
    assert np.isclose(res.sis.prevalence[-1], np.mean([s.results.sis.prevalence[-1] for s in sims])) # Proportions are averaged
    assert np.array_equal(res.bysex.n_infected[-1], sum(s.results.bysex.n_infected[-1] for s in sims)) # Including for 2D results
    msim.plot()

    # Compare
    df = msim.compare(output=True)
    assert df.shape[1] == 3 and 'bysex_n_infected_male' in df.index # One column per sim, one row per result
    assert msim.compare(t=-1, output=True).loc['sis_cum_infections'].tolist() == [s.results.sis.cum_infections[-1] for s in sims]
    msim.compare(t='2005-01-01', do_plot=True)

    # Merge and split
    m1, m2 = msim.split(chunks=[1,2])
    merged = ss.MultiSim.merge(m1, m2)
    assert len(m1) == 1 and len(m2) == 2 and len(merged) == 3
    assert np.array_equal(merged.results.sis.n_I.values, msim.results.sis.n_I.values)
    assert [len(m) for m in merged.split()] == [1, 2] # Uses the chunks from the merge
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
    msim2 = test_multisim_construction(do_plot)
    msim3 = test_combine_merge()
    s3,s4 = test_other()

    T.toc()

    if do_plot:
        plt.show()
