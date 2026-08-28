"""
Test the Loop class
"""

# %% Imports and settings
import gc
import weakref
import pandas as pd
import sciris as sc
import starsim as ss

sc.options(interactive=False) # Assume not running interactively

pars = sc.objdict(
    dur      = ss.years(20),
    n_agents = 1000,
    diseases = 'sis',
    networks = 'random',
)

small_pars = sc.objdict(
    dur = 5,
    n_agents = 100,
    diseases = 'sis',
    networks = 'random',
    rand_seed = 1,
)


@sc.timer()
def test_run_options():
    sc.heading('Testing run options...')
    s1 = ss.Sim(pars).init()
    s2 = ss.Sim(pars).init()

    # Automatic run
    s1.run()

    # Manual run
    while s2.loop.index < len(s2.loop):
        s2.run_one_step()
    s2.finalize()

    assert s1.summary == s2.summary, 'Sims do not match'

    return s2.loop


@sc.timer()
def test_loop_plotting():
    sc.heading('Testing loop plotting...')
    sim = ss.Sim(pars).run(profile=True) # profile=True populates cpu_time for plot_cpu
    assert len(sim.loop.cpu_time), 'Profiling should record per-entry CPU times'
    sim.loop.plot()
    sim.loop.plot_cpu()
    sim.loop.plot_step_order()

    # A non-profiled run should still produce a usable plan DataFrame, just without cpu_time
    sim2 = ss.Sim(pars).run()
    assert not len(sim2.loop.cpu_time), 'Default run should not record CPU times'
    df = sim2.loop.to_df()
    assert 'cpu_time' in df.columns and df['cpu_time'].isna().all(), 'cpu_time should be NaN without profiling'
    return sim.loop


@sc.timer()
def test_memory_cleanup():
    """ Check split-run continuation and sim collectability without shrink(). """
    sc.heading('Testing run memory cleanup...')

    # Partial-run continuation should match an uninterrupted run
    s1 = ss.Sim(small_pars).run()
    s2 = ss.Sim(small_pars)
    s2.run(until=1)
    s2.run()
    assert s1.summary == s2.summary, 'Split run does not match uninterrupted run'

    # Partial and completed sims should be collectable without shrink()
    sim = ss.Sim(small_pars)
    sim.run(until=1)
    ref = weakref.ref(sim)
    del sim
    gc.collect()
    assert ref() is None, 'Partially run sim was not collectable'

    sim = ss.Sim(small_pars).run()
    ref = weakref.ref(sim)
    del sim
    gc.collect()
    assert ref() is None, 'Completed sim was not collectable'

    return


@sc.timer()
def test_callable_cache_cleanup():
    """ Check callable distribution caches are cleared and safely regenerated. """
    sc.heading('Testing callable distribution cache cleanup...')

    def custom_duration(module, sim, uids):
        return 5 + sim.people.age[uids] * 0

    pars = sc.objdict(
        dur = 5,
        n_agents = 100,
        diseases = ss.SIR(init_prev=0.2, dur_inf=ss.normal(loc=custom_duration, scale=0.1)),
        networks = 'random',
        rand_seed = 2,
    )

    # Clearing callable caches after a partial run should not change continuation results
    s1 = ss.Sim(pars).run()
    s2 = ss.Sim(pars)
    s2.run(until=1)
    for dist in s2.dists.dists.values():
        assert getattr(dist, '_callable_args', None) is None
        assert getattr(dist, '_callable_keys', None) is None
    s2.run()
    assert s1.summary == s2.summary, 'Callable cache regeneration changed results'

    # The cleanup finally block should run even when the loop raises
    sim = ss.Sim(pars).init()

    def fail_after_cache(sim):
        dist = next(iter(sim.dists.dists.values()))
        dist._callable_args = {'sim': sim}
        dist._callable_keys = ['sim']
        raise RuntimeError('intentional loop failure')

    sim.loop.insert(fail_after_cache, label='sim.start_step')
    try:
        sim.run()
    except RuntimeError as err:
        assert 'intentional loop failure' in str(err)
    else:
        raise AssertionError('Expected loop failure was not raised')

    for dist in sim.dists.dists.values():
        assert getattr(dist, '_callable_args', None) is None
        assert getattr(dist, '_callable_keys', None) is None
    del dist

    ref = weakref.ref(sim)
    del sim
    gc.collect()
    assert ref() is None, 'Exception-path sim was not collectable'

    return


@sc.timer()
def test_loop_plan_views_and_insert():
    """ Check Python loop storage, dataframe views, and insertion matching. """
    sc.heading('Testing loop plan views and insertions...')

    sim = ss.Sim(small_pars).init()
    assert not isinstance(sim.loop.plan, pd.DataFrame)

    df = sim.loop.to_df()
    assert 'func' not in df.columns
    assert sim.loop.df is df
    assert sim.loop.cpu_df is not None

    calls = []

    def mark_label(sim):
        calls.append(sim.ti)

    sim.loop.insert(mark_label, label='sim.finish_step', before=True)
    sim.run()
    assert len(calls), 'Label-based insertion did not run'

    calls = []
    sim = ss.Sim(small_pars).init()

    def mark_match(sim):
        calls.append(sim.ti)

    def match_fn(plan):
        return plan.label == 'sim.finish_step'

    sim.loop.insert(mark_match, match_fn=match_fn, before=True)
    sim.run()
    assert len(calls), 'Function-based insertion did not run'

    return sim.loop


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    # Run tests
    l1 = test_run_options()
    l2 = test_loop_plotting()
    l3 = test_memory_cleanup()
    l4 = test_callable_cache_cleanup()
    l5 = test_loop_plan_views_and_insert()

    T.toc()
