"""
Test the Loop class
"""

# %% Imports and settings
import gc
import pytest
import numpy as np
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

    # A non-profiled run should still produce a usable plan DataFrame, just without cpu_time (test_loop_plotting() profiles one)
    assert not len(s1.loop.cpu_time), 'Default run should not record CPU times'
    df = s1.loop.to_df()
    assert 'cpu_time' in df.columns and df['cpu_time'].isna().all(), 'cpu_time should be NaN without profiling'
    assert df.time.dtype == float # Plain numbers, not durations (it showed days(0), day, days(2))

    # One time index during a step: every module sees the same one, the step's, where its results go (the first step is ti == 1)
    seen = sc.objdict(rate=[], intv=[])
    def rate(sim):
        seen.rate.append(sim.ti)
        return 0.1
    def intv(sim):
        seen.intv.append(sim.ti)
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, rate=rate), init_prev=0.1, name='sir')
    sim = ss.Sim(sir, interventions=intv, solver='abm', n_agents=100, unit='day', stop=4, verbose=0).run()
    assert seen.intv == [1, 2, 3, 4] and set(seen.rate) == {1, 2, 3, 4}
    assert sim.ti == 4 # At the last time point, after the run

    return s2.loop


@sc.timer()
def test_loop_plotting():
    sc.heading('Testing loop plotting...')
    sim = ss.Sim(pars).run(profile=True) # profile=True populates cpu_time for plot_cpu
    assert len(sim.loop.cpu_time), 'Profiling should record per-entry CPU times'
    sim.loop.plot()
    sim.loop.plot_cpu()
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
    with pytest.raises(RuntimeError, match='intentional loop failure'):
        sim.run()

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
    sim.loop.insert(lambda sim: calls.append(sim.ti), label='sim.finish_step', before=True)
    sim.run()
    assert len(calls) == sim.t.npts - 1, 'Label-based insertion did not run once per step' # Once, not twice

    calls = []
    sim = ss.Sim(small_pars).init()
    sim.loop.insert(lambda sim: calls.append(sim.ti), match_fn=lambda plan: plan.label == 'sim.finish_step', before=True)
    sim.run()
    assert len(calls), 'Function-based insertion did not run'

    return sim.loop


@sc.timer()
def test_coded_alone():
    sc.heading('Testing that a module written as code runs with no agents, and the loop calls only what does something')
    class Decay(ss.Module): # An ODE of its own, with nothing per person
        def __init__(self):
            super().__init__()
            self.y = 1.0
        def step(self):
            self.y *= np.exp(-0.1)
    sim = ss.Sim(modules=Decay(), stop=10, verbose=0).run()
    assert sim.solver != 'abm' and len(sim.people) == 1, 'Code with nothing per person should not make agents'
    assert np.isclose(sim.custom.decay.y, np.exp(-1))
    assert set(sim.loop.to_df().label) == {'sim.advance', 'sim.start_step', 'decay.step', 'sim.finish_step'} # Nothing else does anything here
    assert not sim.check_method_calls() # What was left out counts as called
    return sim


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
    l6 = test_coded_alone()

    T.toc()
