"""
Test results: export, resampling, plotting, 2D results, and nested results
"""
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss
import starsim.library as ssl
import matplotlib.pyplot as plt

sc.options.interactive = False # Assume not running interactively

small = 100
medium = 1000


# %% Define the tests


def test_groups_kept(monkeypatch, tmp_path):
    sc.heading('Testing results kept by group: only those named, and every few steps if every step would take too much memory')
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ 0.3 * I, recovery=I >> R @ ss.days(10), share=I / (S + I + R), init=dict(I=0.01), name='sir')
    make = lambda solver, **kw: ss.Sim(sir, unit='day', stop=91, total_pop=10_000, n_agents=medium, solver=solver, stratify=dict(age=[0, 50]), verbose=0, **kw)
    res = make('ode', results=['I', 'infection']).run().results.sir
    assert res.I.strata is not None and getattr(res.S, 'strata', None) is None and res.infection.cum.strata is not None # Only the named results by group (and the cumulative total of a flow kept by group)
    with pytest.raises(sc.KeyNotFoundError):
        make('ode', results=['J']).init()
    for solver in ['ode', 'binomial', 'abm']: # A whole run in parts, and a stepped run
        full = make(solver).run().results.sir
        monkeypatch.setattr(ss.solvers.Engine, 'group_bytes', 1000) # Too little to keep them by group at every step
        sim = make(solver).run()
        monkeypatch.undo()
        res = sim.results.sir
        k = res.I.group_ti[1] # The steps in each row
        assert k > 1 and any('kept by group every' in note for note in sim.notes)
        assert np.allclose(res.I, full.I) # The totals at every step (as agents, with the same seed)
        assert np.isclose(res.infection.select(age='0-49')[2], full.infection.select(age='0-49')[k+1:2*k+1].sum()) # A flow by group: its total over the steps of a row
        assert np.isclose(res.I.select(age='50+')[2], full.I.select(age='50+')[2*k]) # A state by group: at the end of a row
        assert np.isclose(res.share.select(age='0-49')[-1], full.share.select(age='0-49')[-1]) # A ratio by group: of its parts at the end
        assert np.allclose(res.infection.resample(ss.days(2*k)).select(age='0-49'), full.infection.resample(ss.days(2*k)).select(age='0-49'))

    # Data by group, even from a file, keeps them by group at every step, since it is compared with each one
    sc.dataframe(t=[30.0, 60.0], **{'I[age=0-49]': [10.0, 20.0]}).to_csv(tmp_path/'data.csv', index=False)
    monkeypatch.setattr(ss.solvers.Engine, 'group_bytes', 1000)
    assert make('ode', data=tmp_path/'data.csv').run().results.sir.I.group_ti is None
    monkeypatch.undo()

    # Many runs: a ratio by group is the ratio of its parts in each run, as in one sim (before: the numerator, or an error)
    for solver in ['binomial', 'abm']:
        msim = make(solver).run(n_runs=3)
        share = msim.results.sir.share.select(age='0-49')
        assert np.isclose(share.values[-1, -1], msim.sims[-1].results.sir.share.select(age='0-49')[-1]) and share.values.max() <= 1 # The last run's sim

    # Many runs: by group every few steps too, if all of them would take too much memory
    monkeypatch.setattr(ss.solvers.Compartmental, 'max_bytes', 1000)
    n_I = make('binomial').run(n_runs=3).results.sir.I
    young, old = n_I.select(age='0-49').values, n_I.select(age='50+').values
    assert young.shape == (3, len(n_I.group_ti)) and np.allclose(young + old, n_I.values[:, n_I.group_ti])
    return sim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    # Start timing
    T = sc.tic()

    # Run tests

    sc.toc(T)
    plt.show()
    print('Done.')
