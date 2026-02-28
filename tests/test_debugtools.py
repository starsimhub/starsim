"""
Test Starsim debugging features
"""
import numpy as np
import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss
import pytest

sc.options.interactive = False # Assume not running interactively
# ss.options.warnings = 'error' # Uncomment for additional debugging

small = 100
medium = 1000

debug_pars = dict(
    n_agents = small,
    start = '2000-01-01',
    stop = '2010-01-01',
    diseases = 'sis',
    networks = 'random'
)

# %% Define the tests

@sc.timer()
def test_profile():
    sc.heading('Testing sim.profile()')

    # Based on advanced_profiling.ipynb
    sim = ss.Sim(**debug_pars)
    prof = sim.profile()
    return prof


@sc.timer()
def test_diagnostics_rvs():
    sc.heading('Testing sim diagnostics for random numbers')
    sim = ss.Sim(**debug_pars)
    sim.set_diagnostics(states=False)
    sim.run()

    rvs = sim.diagnostics.rvs
    key = 'diseases_sis_trans_rng_dists_0' # Should be unique every time
    hashes = [entry[key].hash for entry in rvs.values()]
    assert len(hashes) == len(set(hashes))
    return rvs


@sc.timer()
def test_diagnostics_states():
    sc.heading('Testing sim diagnostics for states')
    sim = ss.Sim(**debug_pars, demographics=True)
    sim.set_diagnostics(states=True, detailed=True)
    sim.run()

    states = sim.diagnostics.states
    assert len(states[-1].alive) == len(sim.people)
    return states


@sc.timer()
def test_check_requires():
    sc.heading('Testing check_requires')
    s1 = ss.Sim(diseases='sis', networks='random', n_agents=medium).init()
    ss.check_requires(s1, 'sis')
    ss.check_requires(s1, ss.SIS)
    with pytest.raises(AttributeError):
        ss.check_requires(s1, ss.SIR)
    return s1


@sc.timer()
def test_mock_objects():
    sc.heading('Testing mock objects')
    o = sc.objdict()

    μ = 5
    mock = 100
    atol = 0.1

    # Initializing Dists
    o.dist = ss.lognorm_ex(μ, 2, mock=mock)
    assert np.isclose(o.dist.rvs(1000).mean(), μ, atol=atol), f'Distribution did not have expected mean value of {μ}'

    # Initializing Arrs -- can use it, but cannot grow it since mock slots can't grow
    o.arr = ss.FloatArr('my_arr', default=o.dist, mock=mock)
    assert len(o.arr) == mock, 'Arr did not have expected length of {mock}'
    assert np.isclose(o.arr.mean(), μ, atol=atol), f'Arr did not have expected mean value of {μ}'

    # Initializing People
    o.ppl = ss.People(n_agents=mock, mock=True)
    assert o.ppl.age.mean() > 15, 'People did not have the expected age'

    # Initializing Modules
    o.mod = ss.SIS().init_mock()
    for i in range(5):
        o.mod.step()
        o.mod.set_outcomes(ss.uids([i])) # Manually "infect" people
        o.mod.update_results()

    return o


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    # Start timing
    T = sc.tic()

    # Run tests
    prof = test_profile()
    rvs = test_diagnostics_rvs()
    states = test_diagnostics_states()
    sim = test_check_requires()
    objs = test_mock_objects()

    sc.toc(T)
    plt.show()
    print('Done.')
