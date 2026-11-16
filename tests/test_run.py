"""
Test runs of many sims: ss.MultiSim and ss.parallel()
"""
import pytest
import starsim as ss
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
import corpus

sc.options(interactive=False) # Assume not running interactively


def make_sim(beta=0.1, **kwargs):
    sis = ss.SIS(beta=ss.peryear(beta), dur_inf=ss.years(10)) # In years, as the sim is
    return ss.Sim(sis, n_agents=1000, start=2000, dur=ss.years(10), demographics=True, networks='random', **kwargs)


def make_sir():
    S, I, R = ss.states(3)
    return ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), init_prev=0.01, name='sir')


def test_parallel():
    """ Identical sims match, and differ when they differ; workers started by forkserver give the same runs as by default """
    msim = ss.MultiSim([make_sim(label='Sim1'), make_sim(label='Sim2')])
    msim.run(shrink=False)
    s1, s2 = msim.sims
    assert np.allclose(s1.summarize().value, s2.summarize().value, rtol=0, atol=0, equal_nan=True), "Sims don't match and should"
    s1, s2 = make_sim(), make_sim(beta=0.2)
    ss.parallel(s1, s2, inplace=False)
    assert not s1.initialized
    ss.parallel(s1, s2, inplace=True)
    assert not np.allclose(s1.summarize().value, s2.summarize().value, rtol=0, atol=0, equal_nan=True), "Sims do match and shouldn't"

    make = lambda: [make_sim(interventions=lambda sim: None, rand_seed=i) for i in range(2)] # Once Numba's or JAX's threads have started, by forkserver; with a lambda in the sim
    fork, server = [ss.parallel(make(), par_args=dict(start_method=method)).sims for method in [ss.solvers.backends.workers.start_method(), 'forkserver']]
    assert all(np.array_equal(a.results.sis.prevalence, b.results.sis.prevalence) for a,b in zip(fork, server))
    return s1, s2


def test_restarted():
    """ Many runs of a compartmental sim restart one built sim, from the swept starting values, each with its own seed """
    sir = make_sir()
    msim = ss.parallel(ss.Sim(sir, unit='day', stop=30, total_pop=1000, solver='binomial', verbose=0), n_runs=4)
    assert len(msim.sims.runners) == 1
    sweep = ss.parallel(ss.Sim(sir, total_pop=10_000, stop=30, verbose=0), pars=dict(init_prev=[0.001, 0.01])) # A restart once kept init_prev as it was
    assert np.allclose(sweep.results.sir.I.values[:, 0], [10, 100])

    for kw in [{}, dict(interventions=lambda sim: None)]: # Restarted, or built (with a function in its loop)
        draws = ss.parallel(ss.Sim(sir, solver='binomial', total_pop=10_000, stop=50, verbose=0, **kw), priors=dict(beta=[0.3]*3), workers=1)
        assert isinstance(draws.sims, ss.run.RestartedSims) == (not kw) and list(draws.runs.rand_seed) == [1, 2, 3] # They all had seed 1
        assert len(set(draws.results.sir.infection.values.sum(axis=1))) == 3
        assert draws.sims[2].pars.rand_seed == 3 and np.array_equal(draws.sims[2].results.sir.I.values, draws.results.sir.I.values[2])
    S, I, R = ss.states(3)
    drawn = ss.Disease(infection=S >> I @ ss.par(0.3, bounds=[0.2, 0.4])*I, recovery=I >> R @ ss.par(0.1, prior=ss.normal(0.1, 0.01)), init_n=dict(I=10), name='sir')
    draws = ss.fitting.prior_draws(ss.Sim(drawn, total_pop=1000, stop=5, unit='day', solver='ode', verbose=0), n=8, rand_seed=1)
    assert sorted(np.floor((np.array(list(draws.values())[0]) - 0.2)/0.2*8)) == list(range(8)) # A Latin hypercube: one draw in each slice of the bounds
    fast = ss.parallel(ss.Sim(sir, solver='binomial', total_pop=10_000, stop=50, verbose=0), n_runs=3, workers=1) # Runs taken in the kernels have the sim's own results
    slow = ss.parallel(*[ss.Sim(sir, solver='binomial', total_pop=10_000, stop=50, rand_seed=seed, verbose=0) for seed in [1, 2, 3]], workers=1)
    assert list(fast.results.keys()) == list(slow.results.keys())
    assert all(np.array_equal(fast.results.alive.values[i], slow.sims[i].results.alive.values) for i in range(3))
    sweep = ss.parallel(ss.Sim(sir, solver='binomial', total_pop=10_000, stop=50, verbose=0), pars=dict(beta=[0.3, 0.3, 0.4, 0.4], rand_seed=[1, 2, 1, 2]))
    assert list(sweep.runs.scenario) == ['beta=0.3']*2 + ['beta=0.4']*2 and list(sweep.runs.rand_seed) == [1, 2, 1, 2] # Each seed was a scenario of its own
    return


def test_multisim():
    """ MultiSim results have a run axis first, with summaries over the runs, also for 2D results and sims already run; a subclass changes each run """
    assert len(ss.MultiSim(make_sim(), workers=1).run()) == 1 # One sim is one run, unless n_runs= says more
    msim = ss.MultiSim(make_sim(analyzers=corpus.BySex()), n_runs=4).run()
    res = msim.results.sis.I
    npts = msim.sims[0].t.npts
    assert len(msim) == 4 and isinstance(res, ss.Result) and res.shape == (4, npts)
    assert np.array_equal(res.select(run=2).values, msim.sims[2].results.sis.I.values)
    msim.plot() # The median and an interval
    msim.plot(lines=True) # Every run
    low, mid, high = res.quantiles([0.1, 0.5, 0.9])
    assert np.all(low <= mid) and np.all(mid <= high) and np.array_equal(res.median().values, np.median(res.values, axis=0))
    assert msim.summarize().loc['sis.I', 'std'] > 0
    assert 'Number of runs: 4' in msim.show()
    two = msim.results.sis.S.resample(ss.years(2))
    assert np.array_equal(two.values[:, 1], msim.results.sis.S.values[:, 2]) # A stock at the start of each two years
    df = msim.to_df(quantiles=None)
    assert len(df[(df.module == 'sis') & (df.result == 'I')]) == 4*npts # One row per run and time

    res = msim.results.bysex.infected # 2D results keep their columns
    assert res.shape == (4, npts, 2) and res.median().shape == (npts, 2) and np.array_equal(res.median().male, np.median(res.values[..., 1], axis=0))
    assert 'infected_male' in set(msim.to_df().result)

    msim3 = ss.MultiSim(msim.sims) # Sims that have already run
    assert np.array_equal(msim3.results.sis.I.values, msim.results.sis.I.values)

    class Doubled(ss.MultiSim):
        run_options = ss.MultiSim.run_options + ['beta']
        def make_run(self, sim, values, seed, beta=0.1):
            return super().make_run(sim, dict(values or {}, beta=beta*(1 + seed % 2)), seed)
    doubled = Doubled(make_sim(), n_runs=2, beta=0.05).run(workers=1)
    assert sorted(float(s.diseases.sis.pars.beta.value) for s in doubled.sims) == [0.05, 0.1]
    plt.close('all')
    return msim


def test_scenarios():
    """ Scenarios: by label, their colors in plots, their summaries, compared with a baseline, a result only one has, and sims with different steps """
    sir = make_sir()
    agents = dict(n_agents=300, dur=ss.days(50), verbose=0)
    msim = ss.parallel({'Base': ss.Sim(sir, **agents), 'Vaccine': ss.Sim(sir, interventions=ss.vaccinate(efficacy=0.9, when=20, coverage=0.5, name='vx'), **agents)}, n_runs=10, workers=1)
    assert [k for k,v in msim.sims.items()] == ['Base', 'Vaccine'] and len(msim.sims.items()[0][1]) == 10
    assert list(msim.results.vx.delivered.runs.scenario) == ['Vaccine']*10 # It was left out, so msim.plot('vx.delivered') raised KeyNotFoundError
    fig = msim.plot(lines=True, do_show=False) # Every run of a scenario in its color
    ax_I = next(ax for ax in fig.axes if ax.get_title().endswith(': I'))
    assert len({line.get_color() for line in ax_I.lines}) == 2

    res = msim.results.sir.infection
    assert list(res.summarize().keys()) == ['Base', 'Vaccine'] and len(res.summarize().Base) == 10
    assert msim.summarize('sir.infection').equals(msim.summarize().loc['sir.infection']) and ('vx.delivered', 'Vaccine') in msim.summarize().index
    averted = msim.summarize(baseline='Base').loc[('sir.infection', 'Vaccine')]
    paired = msim.results.sir.infection.diff('Base').Vaccine.summarize()
    assert averted['median'] < 0 and np.isclose(float(averted['median']), np.median(paired), rtol=1e-3) # The paired differences
    sim = msim.sims[0]
    assert sim.summarize('sir.I', how='max') == sim.results.sir.I.max() and msim.summarize('sir.S', how='min').equals(msim.summarize('sir.S')) # Peaks and troughs by name
    assert msim.summarize(how={'sir.I': 'max'}).equals(msim.summarize(how=dict(sir_I='max'))) # A result in how= is named as in key=
    with pytest.raises(KeyError): # Rather than quietly summarized the usual way
        sim.summarize(how=dict(sir_II='max'))

    kw = dict(total_pop=1_000, dur=ss.days(50), solver='binomial', verbose=0)
    once = ss.parallel(ss.Sim(sir, **kw), ss.Sim(sir, **kw, rand_seed=2), workers=1) # Each scenario run once: the same table as a sim's
    for s,(label,table) in zip(once.sims, once.summarize().groupby(level='scenario')):
        assert table.droplevel('scenario').equals(s.summarize()) and 'death' not in s.summarize().index # Nothing in the sim makes deaths, so it has no death result
    swept = ss.parallel(ss.Sim(sir, **kw), pars={'sir.beta': [0.2, 0.3]}, n_runs=2) # Restarted for their runs, labeled by the sweep
    assert [s.label for s in swept.sims] == ['sir.beta=0.2']*2 + ['sir.beta=0.3']*2
    for m in [msim, swept]: # The results in the order of a sim's, however they were run
        assert list(dict.fromkeys(m.summarize().index.get_level_values(0)))[:3] == ['sir.S', 'sir.I', 'sir.R']

    steps = ss.parallel(ss.Sim(sir, **kw), ss.Sim(sir, dt=ss.days(0.5), **kw)) # Sims with different steps: plotted each on its own time axis
    assert {len(line.get_xdata()) for line in steps.plot('sir.I', do_show=False).axes[0].lines} == {51, 101}
    plt.close('all')
    return msim


def test_one_way():
    """ One way to run each thing; results of each sim's one disease; summaries by run """
    sir = make_sir()
    sim = ss.Sim(sir, unit='day', stop=30, total_pop=1000, solver='binomial', verbose=0)
    assert sim.run() is sim and sim.restart(pars=dict(beta=0.4)).run() is sim
    with pytest.raises(TypeError): # Sims are not keywords
        ss.parallel(**dict(baseline=sim))
    msim = ss.parallel({'a': sim.new(), 'b': sim.new(rand_seed=2)}, n_runs=2, workers=1)
    each = msim.summarize(each=True) # A column for each run
    assert list(each.columns) == ['a (0)', 'a (1)', 'b (2)', 'b (3)'] and each.loc['sir.I', 'a (0)'] == msim.sims[0].results.sir.I[-1]
    assert msim.summarize(each=True, t=10).loc['sir.I'].tolist() == [s.results.sir.I[10] for s in msim.sims]

    kw = dict(n_agents=500, dur=ss.days(30), unit='day', verbose=0) # Results across the runs are found in each sim's one disease, even if named differently
    S, E, I, R = ss.states('S E I R')
    sir = ss.Disease(S >> I @ 0.2*I, I >> R @ ss.days(10))
    seir = ss.Disease(S >> E @ 0.2*I, E >> I @ ss.days(3), I >> R @ ss.days(10))
    msim = ss.parallel(ss.Sim(sir, label='SIR', **kw), ss.Sim(seir, label='SEIR', **kw), workers=1, shrink=False)
    assert np.array_equal(msim.results.I.values, [msim.sims[0].results.sir.I.values, msim.sims[1].results.seir.I.values])
    assert msim.results.E.run_pos == [1] # A result of only one of the diseases
    msim.plot(['I', 'infection'])
    plt.close('all')
    several = ss.parallel(ss.Sim([ss.SIR(), ss.SIS()], label='a', **kw), ss.Sim(ss.SIR(), label='b', **kw), workers=1)
    with pytest.raises(AttributeError):
        several.results.I # A sim with several diseases still needs the disease

    kw = dict(diseases='sis', networks='random', n_agents=200, start=2000, stop=2005, verbose=0) # ss.MultiSim(s1, s2) takes several sims (s2 was the label)
    s1, s2 = ss.Sim(**kw, label='a'), ss.Sim(**kw, label='b')
    msim = ss.MultiSim(s1, s2, label='both', workers=1)
    assert len(msim.sims) == 2 and msim.label == 'both'
    summary = msim.run().summarize()
    assert summary.sis.infection.equals(summary.loc['sis.infection']) # Read as attributes
    one = s1.summarize()
    assert one.loc['sis.I', 'value'] == one.value.sis.I == s1.summarize('sis_I') == s1.results.sis.I[-1]
    vx = ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.5, when=[2001, 2003], name='vx') # A delivery's coverage over runs is each run's total delivered over total eligible
    vmsim = ss.parallel(ss.Sim(**kw, interventions=vx), n_runs=2, workers=1)
    each = [sim.summarize('vx.coverage') for sim in vmsim.sims]
    assert np.allclose(vmsim.summarize(each=True).loc['vx.coverage'], each) and np.isclose(vmsim.summarize('vx.coverage')['mean'], np.mean(each))
    return msim


if __name__ == '__main__':
    T = sc.timer()
    test_parallel()
    test_restarted()
    test_multisim()
    test_scenarios()
    test_one_way()
    T.toc()
