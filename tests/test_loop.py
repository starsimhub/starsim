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


def legacy_plan_tuples(loop):
    """
    Reconstruct the integration plan using the original (pre-rc3.6.0) algorithm:
    build every (function × module-time) entry, then sort by the (time, func_order)
    object key. Used to prove the new numeric/uniform construction is identical.
    """
    raw = []
    for fr in loop.funcs:
        module = fr['module']
        func_name = fr['func_name']
        func_order = fr['func_order']
        label = f'{module}.{func_name}'
        for t in loop.abs_tvecs[module]:
            raw.append(dict(time=t, func_order=func_order, label=label, module=module, func_name=func_name))
    raw.sort(key=lambda e: (e['time'], e['func_order'])) # The original object-key sort
    ti = -1
    out = []
    for e in raw:
        if e['label'] == 'sim.start_step':
            ti += 1
        out.append((ti, e['func_order'], e['label'], e['module'], e['func_name'], str(e['time'])))
    return out


def actual_plan_tuples(loop):
    """ Canonical tuple view of the current plan for comparison """
    return [(e.ti, e.func_order, e.label, e.module, e.func_name, str(e.time)) for e in loop.plan]


@sc.timer()
def test_plan_identity():
    """ The rewritten make_plan (uniform fast path + numeric sort) must reproduce the legacy plan exactly. """
    sc.heading('Testing integration plan identity vs. the legacy object-sort...')

    configs = dict(
        # Uniform (fast-path) cases -- all modules share the sim timeline
        uniform_years = dict(diseases='sis', networks='random', demographics=True, dur=10, n_agents=100),
        uniform_dates = dict(diseases=['sir','sis'], networks='random', start='2000-01-01', stop='2003-01-01', dt=ss.days(5), n_agents=100),
        bare          = dict(start=0, stop=365*3, dt=ss.days(1)),
        # Heterogeneous (fallback) cases -- modules with different dt/units
        het_calendar  = dict(diseases=ss.SIS(dt=ss.days(1)), networks=ss.RandomNet(dt=ss.weeks(1)), demographics=ss.Births(dt=ss.days(10)), dt=ss.days(2), start='2000-01-01', stop='2001-01-01', n_agents=100),
        het_relative  = dict(diseases=ss.SIS(dt='month'), networks='random', dur=ss.years(2), dt=ss.years(1/12), n_agents=100),
        het_coincident= dict(diseases=ss.SIS(dt=1.0), networks=ss.RandomNet(dt=2.0), dur=10, dt=1.0, n_agents=100),
        het_sparse    = dict(diseases=ss.SIS(dt=0.1), demographics=ss.Births(dt=2.0), networks='random', dur=10, dt=0.1, n_agents=100),
    )

    uniform_seen = set()
    for name, pars in configs.items():
        sim = ss.Sim(verbose=0, **pars).init()
        legacy = legacy_plan_tuples(sim.loop)
        actual = actual_plan_tuples(sim.loop)
        uniform = sim.loop._timelines_uniform()
        uniform_seen.add(uniform)
        assert actual == legacy, f'Plan mismatch for config "{name}" (uniform={uniform})'
        print(f'  ✓ {name}: {len(actual)} entries match (uniform_fast_path={uniform})')

    # Both the uniform fast path and the heterogeneous (numeric-sort) fallback must be exercised
    assert uniform_seen == {True, False}, f'Expected both plan-construction paths to be tested, got uniform={uniform_seen}'

    return sim.loop


@sc.timer()
def test_plan_identity_with_insertions():
    """ Plan identity must also hold for the base plan after loop.insert() insertions. """
    sc.heading('Testing plan identity with insertions...')

    # Insertions are replayed after the base plan is built; the base ordering must still match legacy
    sim = ss.Sim(small_pars).init()

    def probe(sim):
        pass

    sim.loop.insert(probe, label='sim.finish_step', before=True)

    # Compare the base (non-inserted) rows against the legacy plan; inserted rows have module=None
    legacy = legacy_plan_tuples(sim.loop)
    actual = [t for t in actual_plan_tuples(sim.loop) if t[3] is not None] # Drop inserted rows (module is None)
    assert actual == legacy, 'Base plan mismatch after insertion'

    # Ensure the sim still runs and the inserted function is actually called
    calls = []
    sim2 = ss.Sim(small_pars).init()
    sim2.loop.insert(lambda sim: calls.append(sim.ti), label='sim.finish_step', before=True)
    sim2.run()
    assert len(calls), 'Inserted function did not run'

    return sim.loop


@sc.timer()
def test_skip_noop_update_results():
    """ A module's update_results is skipped only if it's the inherited no-op base method. """
    sc.heading('Testing no-op lifecycle filtering...')

    def noop_intv(sim): # Function-based intervention: base update_results, no auto states -> skippable
        pass

    class OverrideIntv(ss.Intervention): # Overrides update_results (via super) -> must be kept
        def step(self):
            pass
        def update_results(self):
            super().update_results()

    sim = ss.Sim(diseases='sis', networks='random', n_agents=100, dur=3,
                 interventions=[noop_intv, OverrideIntv()], verbose=0).init()
    labels = [f"{f['module']}.{f['func_name']}" for f in sim.loop.funcs]
    ur = [l for l in labels if l.endswith('update_results')]
    assert 'noop_intv.update_results' not in ur, 'No-op intervention update_results should be skipped'
    assert any('overrideintv' in l for l in ur), 'An override (even one calling super()) must be kept'
    assert 'sis.update_results' in ur, 'A disease that overrides update_results must be kept'
    assert 'people.update_results' in ur, 'People update_results is always kept'

    # Result-identical: skipping a genuinely no-op update_results does not change outcomes
    kw = dict(diseases='sis', networks='random', n_agents=200, dur=5, rand_seed=1, verbose=0)
    s1 = ss.Sim(interventions=noop_intv, **kw).run()
    s2 = ss.Sim(**kw).run()
    assert s1.summary.sis_cum_infections == s2.summary.sis_cum_infections, 'No-op filtering changed results'

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
    l6 = test_plan_identity()
    l7 = test_plan_identity_with_insertions()
    l8 = test_skip_noop_update_results()

    T.toc()
