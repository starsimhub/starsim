"""
Test Sim API
"""

# %% Imports and settings
import sys
import numpy as np
import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss
import pytest

n_agents = 1_000
do_plot = False
sc.options(interactive=False) # Assume not running interactively


def test_single_defaults(do_plot=do_plot):
    """ Test a single MixingPool using defaults """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mp = ss.MixingPool()
    sir = ss.SIR()
    sim = ss.Sim(diseases=sir, networks=mp, label=test_name)
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    return sim


def test_single_uids(do_plot=do_plot):
    """ Test a single MixingPool by UIDS, pre-initialization configuration """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')

    n_agents = 10_000
    k = n_agents // 2
    mp = ss.MixingPool(
        src = ss.uids(np.arange(k)),
        dst = ss.uids(np.arange(k,n_agents))
    )
    sir = ss.SIR()
    sim = ss.Sim(n_agents=n_agents, diseases=sir, networks=mp, label=test_name)
    sim.run()

    if do_plot: sim.plot()

    assert sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0], 'There were no new infections in the simulation'
    return sim


def test_single_ncd():
    """ Test a single MixingPool with a ncd """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mp_pars = {
        'src': ss.AgeGroup(0, 15),
        'dst': ss.AgeGroup(15, None),
        'beta': 1.0,
        'contacts': ss.poisson(lam=5),
        'diseases': 'ncd'
    }
    mp = ss.MixingPool(mp_pars)

    ncd = ss.NCD()
    sim = ss.Sim(diseases=ncd, networks=mp, label=test_name)
    with pytest.raises(Exception):
        sim.run()
    return sim


def test_single_missing_disease():
    """ Test a single MixingPool with a missing disease """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mp_pars = {
        'src': ss.AgeGroup(0, 15),
        'dst': ss.AgeGroup(15, None),
        'beta': 1.0,
        'contacts': ss.poisson(lam=5),
        'diseases': 'hiv'
    }
    mp = ss.MixingPool(mp_pars)

    sir = ss.SIR()
    sim = ss.Sim(diseases=sir, networks=mp, label=test_name)
    with pytest.raises(Exception):
        sim.run()

    return sim


def test_single_age(do_plot=do_plot):
    """ Test a single MixingPool by age """
    # Incidence must decline because 0-15 --> 15+ transmission only
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mp_pars = {
        'src': ss.AgeGroup(0, 15),
        'dst': ss.AgeGroup(15, None),
        'beta': 1.0,
        'contacts': ss.poisson(lam=5),
    }
    mp = ss.MixingPool(mp_pars)

    sir = ss.SIR()
    sim = ss.Sim(diseases=sir, networks=mp, label=test_name)
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    return sim


def test_single_sex(do_plot=do_plot):
    """ Test a single MixingPool by sex """
    # Incidence must decline because M --> F transmission only
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mp_pars = {
        'src': lambda sim: sim.people.female, # female to male (only) transmission
        'dst': lambda sim: sim.people.male,
        'beta': 1.0,
        'contacts': ss.poisson(lam=4),
    }
    mp = ss.MixingPool(mp_pars)

    sir = ss.SIR(init_prev=ss.bernoulli(p=lambda self, sim, uids: 0.05*sim.people.female)) # Seed 5% of the female population
    sim = ss.Sim(diseases=sir, networks=mp, label=test_name)
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    assert(sim.people.male[sim.diseases.sir.ti_infected>0].all()) # All new infections should be in men
    return sim


def test_multi_defaults(do_plot=do_plot):
    """ Test MixingPools using defaults """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mps = ss.MixingPools(src={'all':None}, dst={'all':None}, contacts=[[1]])
    sir = ss.SIR()
    sim = ss.Sim(diseases=sir, networks=mps, label=test_name)
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    return sim


def test_multi(do_plot=do_plot):
    """ Test MixingPools """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')

    groups = {
        'Female': lambda sim: sim.people.female,
        'Male': lambda sim: sim.people.male,
    }

    mps_pars = dict(
        contacts = np.array([[1.4, 0.5], [1.2, 0.7]]),
        beta = 1,
        src = groups,
        dst = groups,
    )
    mps = ss.MixingPools(mps_pars)

    sir = ss.SIR()
    sim = ss.Sim(diseases=sir, networks=mps, label=test_name)
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    return sim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    sim0 = test_single_defaults(do_plot)
    sim1 = test_single_uids(do_plot)
    sim2 = test_single_ncd()
    sim3 = test_single_missing_disease()
    sim4 = test_single_age(do_plot)
    sim5 = test_single_sex(do_plot)

    sim6 = test_multi_defaults(do_plot)
    sim7 = test_multi(do_plot)

    T.toc()

    if do_plot:
        plt.show()
