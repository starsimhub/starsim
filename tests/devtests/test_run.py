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
    pars = sc.obj(
        n_agents = n_agents,
        start = 2000,
        dur = ss.years(10),
        demographics = True,
        networks = 'random',
        diseases = sc.obj(type='sis', beta=beta), # To allow for modification later
    )
    return sc.obj(pars, **kwargs)


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
def test_multisim_construction(do_plot=False):
    """ Test other MultiSim construction methods """
    sc.heading('Testing MultiSim construction')
    
    # Make sims
    def make_sims(n=2):
        sims = sc.autolist()
        for beta in 0.1*np.random.rand(n):
            pars = make_sim_pars(beta=beta, rand_seed=np.random.randint(999))
            sims += ss.Sim(pars, label=f'{beta=}')
        return sims
    
    # Construction options
    m1 = ss.MultiSim(make_sims(), label='list') # As list
    m2 = ss.MultiSim(tuple(make_sims()), label='tuple') # As tuple
    m3 = ss.MultiSim(m1.copy(), label='msim') # As MultiSim
    m4 = ss.MultiSim([m1.copy(), m2.copy()], label='multi-msim') # As list of MultiSims
    
    # Check that run works
    for msim in [m1, m2, m3, m4]:
        msim.run(workers=None if msim is m4 else 1) # Only one in parallel, to save starting workers
        if do_plot:
            msim.plot()
    
    return m4


@sc.timer()
def test_combine_merge():
    """ Test MultiSim combine(), summarize(each=True), merge() and split() """
    sc.heading('Testing MultiSim combine, summarize by run, merge, and split')
    msim = ss.MultiSim(ss.Sim(make_sim_pars(analyzers=BySex())), n_runs=3)
    msim.run(workers=1)
    sims = msim.sims

    # Combine
    sim = msim.combine()
    res = sim.results
    assert sim.pars.n_agents == 3*n_agents and sim.pop_scale == 1 # Population sizes are summed
    assert res.sis.I[-1] == sum(s.results.sis.I[-1] for s in sims) # Counts are summed
    assert np.isclose(res.sis.prevalence[-1], np.mean([s.results.sis.prevalence[-1] for s in sims])) # Proportions are averaged
    assert np.array_equal(res.bysex.infected[-1], sum(s.results.bysex.infected[-1] for s in sims)) # Including for 2D results
    msim.plot()

    # Each run side by side
    df = msim.summarize(each=True)
    assert df.shape[1] == 3 and 'bysex_infected_male' in df.index # One column per run, one row per result
    assert msim.summarize(each=True, t=-1).loc['sis_I'].tolist() == [s.results.sis.I[-1] for s in sims]
    assert msim.summarize('sis_I', each=True, t='2005-01-01').shape == (3,)
    assert msim.summarize('sis_I', t=-1)['median'] == np.median([s.results.sis.I[-1] for s in sims]) # The statistics across runs at one time point

    # Merge and split
    m1, m2 = msim.split(chunks=[1,2])
    m1.label, m2.label = 'first', 'second'
    merged = ss.MultiSim.merge(m1, m2)
    assert [m.label for m in merged.split()] == ['first', 'second'] # Each part keeps its own label (both were the first's)
    steps = ss.parallel(ss.Sim(diseases='sis', n_agents=200, dt=1, verbose=0), ss.Sim(diseases='sis', n_agents=200, dt=0.5, verbose=0), workers=1) # Sims with different steps are run, but not put together as runs
    with pytest.raises(ValueError):
        steps.summarize()
    assert len(m1) == 1 and len(m2) == 2 and len(merged) == 3
    assert np.array_equal(merged.results.sis.I.values, msim.results.sis.I.values)
    assert [len(m) for m in merged.split()] == [1, 2] # Uses the chunks from the merge
    return msim


@sc.timer()
def test_quiet_quantiles():
    """ Runs of a quiet sim print nothing, and the quantiles of several scenarios unpack by quantile """
    sc.heading('Testing quiet runs and quantiles by scenario')
    kw = dict(diseases='sis', networks='random', n_agents=300, verbose=0)
    with sc.capture() as text:
        ss.parallel(ss.Sim(**kw), n_runs=2)
        msim = ss.parallel(ss.Sim(label='a', **kw), ss.Sim(label='b', rand_seed=5, **kw), n_runs=3, workers=1)
    assert 'MultiSim' not in text and '%' not in text # The sims' verbose=0 is the MultiSim's default

    I = msim.results.sis.I
    low, mid, high = I.quantiles([0.1, 0.5, 0.9]) # A result for each quantile, with a row per scenario
    assert low.values.shape == (2, len(I.timevec))
    assert np.allclose(high.select(scenario='b').values, I.select(scenario='b').quantiles([0.9]).values)
    assert (low.values <= high.values).all()
    return msim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    msim2 = test_multisim_construction(do_plot)
    msim3 = test_combine_merge()
    msim4 = test_quiet_quantiles()

    T.toc()

    if do_plot:
        plt.show()
