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
        diseases = sc.obj(type='sis', beta=ss.peryear(beta), dur_inf=ss.years(10)), # To allow for modification later; in years, as the sim is
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
def test_parallel():
    """ Test running two identical sims in parallel """
    sc.heading('Testing parallel...')
    pars = make_sim_pars()

    # Check that two identical sims match
    msim = ss.MultiSim([ss.Sim(pars, label='Sim1'), ss.Sim(pars, label='Sim2')])
    msim.run(shrink=False)
    s1, s2 = msim.sims
    assert np.allclose(s1.summarize().value, s2.summarize().value, rtol=0, atol=0, equal_nan=True), "Sims don't match and should" # That non-identical sims don't is checked in test_other()

    # Workers started by forkserver, as they are once Numba's or JAX's threads have started here, give the same runs as by default (fork on Linux, if safe), with a lambda in the sim
    make = lambda: [ss.Sim(make_sim_pars(interventions=lambda sim: None), rand_seed=i) for i in range(2)]
    fork, server = [ss.parallel(make(), par_args=dict(start_method=method)).sims for method in [ss.kernels.start_method(), 'forkserver']]
    assert all(np.array_equal(a.results.sis.prevalence, b.results.sis.prevalence) for a,b in zip(fork, server)) # The same either way

    return s1, s2


@sc.timer()
def test_restarted():
    """ Many runs of a compartmental sim restart one built sim, rather than building each """
    sc.heading('Testing that ss.parallel(sim, n_runs=) restarts one sim')
    S, I, R = ss.states(3)
    sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(8), init=dict(I=0.01), name='sir')
    msim = ss.parallel(ss.Sim(sir, unit='day', stop=30, total_pop=1000, solver='binomial', verbose=0), n_runs=4)
    assert len(msim.sims.runners) == 1

    # A restart starts from the swept starting values, including init_prev (which a restart once kept as it was)
    sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(8), init_prev=0.001, name='sir')
    sweep = ss.parallel(ss.Sim(sir, total_pop=10_000, stop=30, verbose=0), pars=dict(init_prev=[0.001, 0.01]))
    assert np.allclose(sweep.results.sir.I.values[:, 0], [10, 100])
    return


def test_multisim():
    """ Check MultiSim methods """
    sc.heading('Testing MultiSim')
    pars = make_sim_pars()
    assert len(ss.MultiSim(ss.Sim(pars), workers=1).run()) == 1 # One sim is one run, unless n_runs= says more
    msim = ss.MultiSim(ss.Sim(pars), n_runs=4)
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
    text = msim.show() # Text that shows itself, as describe() returns
    assert isinstance(text, ss.display.Text) and 'Number of runs: 4' in text
    with pytest.raises(TypeError, match='text = msim.show()'):
        msim.show(output=True) # Starsim 3's way to get the text

    # Export results
    two = msim.results.sis.S.resample(ss.years(2))
    assert np.array_equal(two.values[:, 1], msim.results.sis.S.values[:, 2]) # A stock at the start of each two years from the start
    df = msim.to_df(quantiles=None)
    assert len(df[(df.module == 'sis') & (df.result == 'I')]) == 4*npts # One row per run and time

    # 2D results
    msim2 = ss.MultiSim(ss.Sim(make_sim_pars(analyzers=BySex())), n_runs=3)
    msim2.run(workers=1)
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
    doubled = Doubled(ss.Sim(pars), n_runs=2, beta=0.05).run(workers=1)
    assert sorted(float(s.diseases.sis.pars.beta.value) for s in doubled.sims) == [0.05, 0.1]

    # The sims by scenario, and the results of a module only one scenario has
    kw = dict(n_agents=300, stop=30, verbose=0)
    vx = ss.parallel(ss.Sim(ss.SIR(), label='Base', **kw), ss.Sim(ss.SIR(), interventions=ss.vaccinate(efficacy=0.8, when=[10], coverage=0.5, name='vx'), label='Vaccine', **kw), workers=1)
    assert vx.sims['Vaccine'] is vx.sims[1]
    assert list(vx.results.vx.delivered.runs.scenario) == ['Vaccine'] # It was left out, so msim.plot('vx.delivered') raised KeyNotFoundError
    vx.plot('vx.delivered')
    assert ('vx_delivered', 'Vaccine') in vx.summarize().index
    plt.close('all')

    # Each draw has a seed of its own, restarted or built (with a function in its loop); seeds swept are runs of their scenario
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
    for kw in [{}, dict(interventions=lambda sim: None)]:
        draws = ss.parallel(ss.Sim(sir, solver='binomial', total_pop=10_000, stop=50, verbose=0, **kw), priors=dict(beta=[0.3]*3), workers=1)
        assert isinstance(draws.sims, ss.run.RestartedSims) == (not kw) and list(draws.runs.rand_seed) == [1, 2, 3] # They all had seed 1
        assert len(set(draws.results.sir.infection.values.sum(axis=1))) == 3 # So they differ by chance as well as by their values
        assert draws.sims[2].pars.rand_seed == 3 and np.array_equal(draws.sims[2].results.sir.I.values, draws.results.sir.I.values[2])
    drawn = ss.Disease(infection=S >> I @ ss.par(0.3, bounds=[0.2, 0.4])*I, recovery=I >> R @ ss.par(0.1, prior=ss.normal(0.1, 0.01)), init_n=dict(I=10), name='sir')
    draws = ss.fitting.prior_draws(ss.Sim(drawn, total_pop=1000, stop=5, unit='day', solver='ode', verbose=0), n=8, rand_seed=1)
    assert sorted(np.floor((np.array(list(draws.values())[0]) - 0.2)/0.2*8)) == list(range(8)) # A Latin hypercube: one draw in each slice of the bounds
    # Runs taken in the kernels have the sim's own results, as runs one at a time do
    fast = ss.parallel(ss.Sim(sir, solver='binomial', total_pop=10_000, stop=50, verbose=0), n_runs=3, workers=1)
    slow = ss.parallel(*[ss.Sim(sir, solver='binomial', total_pop=10_000, stop=50, rand_seed=seed, verbose=0) for seed in [1, 2, 3]], workers=1)
    assert list(fast.results.keys()) == list(slow.results.keys())
    assert all(np.array_equal(fast.results.alive.values[i], slow.sims[i].results.alive.values) for i in range(3))
    sweep = ss.parallel(ss.Sim(sir, solver='binomial', total_pop=10_000, stop=50, verbose=0), pars=dict(beta=[0.3, 0.3, 0.4, 0.4], rand_seed=[1, 2, 1, 2]))
    assert list(sweep.runs.scenario) == ['beta=0.3']*2 + ['beta=0.4']*2 and list(sweep.runs.rand_seed) == [1, 2, 1, 2] # Each seed was a scenario of its own

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
    assert not np.allclose(s1.summarize().value, s2.summarize().value, rtol=0, atol=0, equal_nan=True), "Sims do match and shouldn't"

    # Scenarios of a stratified compartmental model restarted for their runs, one with a delivery that adds groups (it raised a ValueError), compared with the baseline, and plotted with weekly data
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), init_n=dict(I=5), name='sir').stratify(age={0: 0.3, 15: 0.7})
    kw = dict(total_pop=10_000, dur=ss.days(100), solver='binomial', verbose=0)
    weekly = ss.Sim(sir, **kw).run().to_synthetic_data('sir.infection', resample='week') # Whole weeks only, as resample() keeps: not the last of the 101 steps', which is three days
    msim = ss.parallel({'Base': ss.Sim(sir, data=weekly, **kw), 'Vaccine': ss.Sim(sir, interventions=ss.vaccinate(efficacy=0.9, when=20, coverage=0.5), **kw)}, n_runs=3)
    averted = msim.summarize(baseline='Base').loc[('sir_infection', 'Vaccine')]
    paired = msim.results.sir.infection.diff('Base').Vaccine.summarize()
    assert averted['median'] < 0 and np.isclose(float(averted['median']), np.median(paired), rtol=1e-3) # The paired differences, as msim.diff() gives
    ax = next(ax for ax in msim.plot(show_interventions=False).axes if 'infection' in ax.get_title())
    assert len(ax.collections) >= 2 and all(len(line.get_xdata()) == len(weekly) for line in ax.lines) # The data, and the model summed over its weeks
    return s1,s2


@sc.timer()
def test_scenarios():
    """ Scenarios of a MultiSim: their colors and labels in plots, their summaries, their sims, and sims with different steps """
    sc.heading('Testing scenarios')
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), init_n=dict(I=5), name='sir')
    kw = dict(total_pop=1_000, dur=ss.days(50), solver='binomial', verbose=0)
    agents = dict(n_agents=300, dur=ss.days(50), verbose=0)
    msim = ss.parallel({'Base': ss.Sim(sir, **agents), 'Vaccine': ss.Sim(sir, interventions=ss.vaccinate(efficacy=0.9, when=20, coverage=0.5), **agents)}, n_runs=10, workers=1)

    # Every run of a scenario in its color, with as many runs as there are colors (they were all one color), and the same color in axes of a result only one scenario has
    fig = msim.plot(lines=True, do_show=False)
    colors = lambda ax: {line.get_label():line.get_color() for line in ax.lines if not line.get_label().startswith('_')}
    ax_I = next(ax for ax in fig.axes if ax.get_title().endswith(': I'))
    ax_vx = next(ax for ax in fig.axes if 'delivered' in ax.get_title())
    assert len({line.get_color() for line in ax_I.lines}) == 2 and colors(ax_vx) == {'Vaccine':colors(ax_I)['Vaccine']}
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ['Base', 'Vaccine']

    # Summaries by scenario, and of one result
    res = msim.results.sir.infection
    assert list(res.summarize().keys()) == ['Base', 'Vaccine'] and len(res.summarize().Base) == 10
    assert msim.summarize('sir.infection').equals(msim.summarize().loc['sir_infection'])
    sim = msim.sims[0]
    assert sim.summarize('sir.infection') == sim.summarize().value.sir_infection
    once = ss.parallel(ss.Sim(sir, **kw), ss.Sim(sir, **kw, rand_seed=2), workers=1) # Each scenario run once: the same table as a sim's, with a value column
    for s,(label,table) in zip(once.sims, once.summarize().groupby(level='scenario')):
        assert table.droplevel('scenario').equals(s.summarize()) and s.summarize().value.died == 0 # Results that are 0 throughout are kept, so the tables line up
    assert sim.summarize('sir.I', how='max') == sim.results.sir.I.max() and msim.summarize('sir.S', how='min').equals(msim.summarize('sir.S')) # Peaks and troughs by name (S only falls, so its least is its last)
    with pytest.raises(ValueError, match="Unknown how='maxx'"): # Rather than N/A in the table
        sim.summarize(how='maxx')
    with pytest.raises(ValueError, match="Unknown how='maxx' for sir_I"):
        msim.summarize(how=dict(sir_I='maxx'))
    assert sim.summarize(how={'sir.I': 'max'}).value.sir_I == sim.results.sir.I.max() # A result in how= is named as in key=, by module and name
    assert msim.summarize(how={'sir.I': 'max'}).equals(msim.summarize(how=dict(sir_I='max')))
    with pytest.raises(KeyError, match='"sir_II" is not a result'): # Rather than quietly summarized the usual way
        sim.summarize(how=dict(sir_II='max'))

    # The sims by scenario, labeled by it when swept
    assert [k for k,v in msim.sims.items()] == ['Base', 'Vaccine'] and len(msim.sims.items()[0][1]) == 10
    swept = ss.parallel(ss.Sim(sir, **kw), pars={'sir.beta': [0.2, 0.3]}, n_runs=2) # Restarted for their runs
    assert [s.label for s in swept.sims] == ['sir.beta=0.2']*2 + ['sir.beta=0.3']*2
    for m in [msim, swept]: # The results in the order of a sim's (the modules' first), however they were run
        assert list(dict.fromkeys(m.summarize().index.get_level_values(0)))[:3] == ['sir_S', 'sir_I', 'sir_R']

    # Sims with different steps: run, and plotted each on its own time axis, but not put together
    steps = ss.parallel(ss.Sim(sir, **kw), ss.Sim(sir, dt=ss.days(0.5), **kw))
    assert not steps._same_times and {len(line.get_xdata()) for line in steps.plot('sir.I', do_show=False).axes[0].lines} == {51, 101}
    with pytest.raises(ValueError):
        steps.results.sir.I.median()
    return msim


@sc.timer()
def test_one_disease():
    """ Results across the runs are found in each sim's one disease, even if the diseases are named differently """
    sc.heading('Testing msim results and plots of each sim\'s one disease')
    kw = dict(n_agents=500, dur=ss.days(30), unit='day', verbose=0)
    S, E, I, R = ss.states('S E I R')
    sir = ss.Disease(S >> I @ 0.2*I, I >> R @ ss.days(10))
    seir = ss.Disease(S >> E @ 0.2*I, E >> I @ ss.days(3), I >> R @ ss.days(10))
    msim = ss.parallel(ss.Sim(sir, label='SIR', **kw), ss.Sim(seir, label='SEIR', **kw), workers=1, shrink=False)
    assert np.array_equal(msim.results.I.values, [msim.sims[0].results.sir.I.values, msim.sims[1].results.seir.I.values]) # It was 'I' not found, and msim.plot('I') needed name=
    assert msim.results.E.run_pos == [1] # A result of only one of the diseases is across its runs
    msim.plot(['I', 'infection'])
    plt.close('all')
    several = ss.parallel(ss.Sim([ss.SIR(), ss.SIS()], label='a', **kw), ss.Sim(ss.SIR(), label='b', **kw), workers=1)
    with pytest.raises(AttributeError):
        several.results.I # A sim with several diseases still needs the disease
    return msim


@sc.timer()
def test_one_way():
    """ One way to run each thing: sim.run() returns the sim, ss.parallel() runs many, and the old spellings say what replaced them """
    sc.heading('Testing the one way to run, summarize, and describe')
    S, I, R = ss.states(3)
    sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(8), init=dict(I=0.01), name='sir')
    sim = ss.Sim(sir, unit='day', stop=30, total_pop=1000, solver='binomial', verbose=0)
    assert sim.run() is sim and sim.restart(pars=dict(beta=0.4)).run() is sim
    for key,new in dict(shrink='sim.shrink', profile="kind='loop'").items(): # Starsim 3's: each names what replaced it
        with pytest.raises(TypeError, match=new):
            sim.run(**{key: 1})
    with pytest.raises(TypeError, match='dict by label'): # Sims are not keywords, so the keywords are only options
        ss.parallel(**dict(baseline=sim)) # As keywords (written so, so that rewrite scripts leave it)
    with pytest.raises(TypeError, match='workers=1'):
        ss.parallel(sim, **dict(parallel=False))
    msim = ss.parallel({'a': sim.new(), 'b': sim.new(rand_seed=2)}, n_runs=2, workers=1)
    each = msim.summarize(each=True) # A column for each run, as compare() gave
    assert list(each.columns) == ['a (0)', 'a (1)', 'b (2)', 'b (3)'] and each.loc['sir_I', 'a (0)'] == msim.sims[0].results.sir.I[-1]
    assert msim.summarize(each=True, t=10).loc['sir_I'].tolist() == [s.results.sir.I[10] for s in msim.sims]

    # Text that shows itself, rather than output=
    text = sim.diseases.sir.equations()
    assert isinstance(text, str) and text.startswith('dS/dt') and isinstance(sim.describe(), str) and sim.check_dt().table is not None
    with pytest.raises(AttributeError, match='summarize'):
        getattr(sim, 'summary') # Removed, for sim.summarize()
    return msim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    s1, s2 = test_parallel()
    msim = test_multisim()
    s3,s4 = test_other()
    msim2 = test_scenarios()
    msim3 = test_one_disease()
    msim4 = test_one_way()

    T.toc()

    if do_plot:
        plt.show()
