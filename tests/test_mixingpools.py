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

pars = sc.objdict(
    n_agents = 1_000
)
do_plot = False
sc.options(interactive=False) # Assume not running interactively


@sc.timer()
def test_single_defaults(do_plot=do_plot):
    """ Test a single MixingPool using defaults """
    sc.heading('Testing single mixing pool...')
    mp = ss.MixingPool()
    sir = ss.SIR()
    sim = ss.Sim(pars, diseases=sir, networks=mp)
    sim.run()
    res = sim.results.sir

    if do_plot: sim.plot()

    assert res.cum_infections[-1] > res.cum_infections[0] # There were infections
    return sim


@sc.timer()
def test_single_uids(do_plot=do_plot):
    """ Test a single MixingPool by UIDS, pre-initialization configuration """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')

    # Note: in this arrangement, src transmits to dst, but can never go the other way, so the epidemic dies out once everyone in src recovers
    k = pars.n_agents // 2
    mp = ss.MixingPool(
        src = ss.uids(np.arange(k)),
        dst = ss.uids(np.arange(k, pars.n_agents))
    )
    sir = ss.SIR()
    sim = ss.Sim(pars, diseases=sir, networks=mp, label=test_name)
    sim.run()

    if do_plot: sim.plot()

    assert sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0], 'There were no new infections in the simulation'
    return sim


@sc.timer()
def test_single_ncd():
    """ Test a single MixingPool with a ncd """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mp_pars = dict(
        src = ss.AgeGroup(0, 15),
        dst = ss.AgeGroup(15, None),
        beta = 1,
        n_contacts = ss.poisson(lam=5),
        diseases = 'ncd'
    )
    mp = ss.MixingPool(**mp_pars)

    ncd = ss.NCD()
    sim = ss.Sim(pars, diseases=ncd, networks=mp, label=test_name)
    with pytest.raises(Exception):
        sim.run()
    return sim


@sc.timer()
def test_single_missing_disease():
    """ Test a single MixingPool with a missing disease """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mp_pars = dict(
        src = ss.AgeGroup(0, 15),
        dst = ss.AgeGroup(15, None),
        beta = 1.0,
        n_contacts = ss.poisson(lam=5),
        diseases = 'hiv'
    )
    mp = ss.MixingPool(**mp_pars)

    sir = ss.SIR()
    sim = ss.Sim(pars, diseases=sir, networks=mp, label=test_name)
    with pytest.raises(Exception):
        sim.run()

    return sim


@sc.timer()
def test_single_age(do_plot=do_plot):
    """ Test a single MixingPool by age """
    # Incidence must decline because 0-15 --> 15+ transmission only
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mp_pars = dict(
        src = ss.AgeGroup(0, 15),
        dst = ss.AgeGroup(15, None),
        beta = 1.0,
        n_contacts = ss.poisson(lam=5),
    )
    mp = ss.MixingPool(**mp_pars)

    sir = ss.SIR()
    sim = ss.Sim(pars, diseases=sir, networks=mp, label=test_name)
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    return sim


@sc.timer()
def test_single_sex(do_plot=do_plot):
    """ Test a single MixingPool by sex """
    # Incidence must decline because M --> F transmission only
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mp_pars = dict(
        src = lambda sim: sim.people.female, # female to male (only) transmission
        dst = lambda sim: sim.people.male,
        beta = 1.0,
        n_contacts = ss.poisson(lam=4),
    )
    mp = ss.MixingPool(**mp_pars)

    sir = ss.SIR(init_prev=ss.bernoulli(p=lambda self, sim, uids: 0.05*sim.people.female)) # Seed 5% of the female population
    sim = ss.Sim(pars, diseases=sir, networks=mp, label=test_name)
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    assert(sim.people.male[sim.diseases.sir.ti_I>0].all()) # All new infections should be in men
    return sim


@sc.timer()
def test_multi_defaults(do_plot=do_plot):
    """ Test MixingPools using defaults """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mps = ss.MixingPools(src={'all':None}, dst={'all':None}, n_contacts=[[1]])
    sir = ss.SIR()
    sim = ss.Sim(pars, diseases=sir, networks=mps, label=test_name)
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    return sim


@sc.timer()
def test_multi(do_plot=do_plot):
    """ Test MixingPools """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')

    groups = {
        'Female': lambda sim: sim.people.female,
        'Male': lambda sim: sim.people.male,
    }

    mps_pars = dict(
        n_contacts = np.array([[1.4, 0.5], [1.2, 0.7]]),
        beta = 1.0,
        src = groups,
        dst = groups,
    )
    mps = ss.MixingPools(**mps_pars)

    sir = ss.SIR()
    sim = ss.Sim(pars, diseases=sir, networks=mps, label=test_name)
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    return sim


@sc.timer()
def test_layers():
    sc.heading('Testing contact layers with a beta for each, under both engines')
    def run(beta, solver, layers=('home', 'school')):
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')
        nets = [ss.contacts('Kenya', layer=layer) for layer in layers]
        sim = ss.Sim(sir, networks=nets, stratify=dict(age=[0, 5, 18, 65]), unit='day', stop=100, solver=solver, total_pop=10_000, verbose=0, **(dict(n_agents=10_000) if solver == 'abm' else {}))
        return sim.run().results.sir.infection.cum[-1]
    ode = run(dict(home=0.03, school=0.02), 'ode') # The spec's layered example, which crashed under both engines
    assert np.isclose(run(dict(home=0.03, school=0.02), 'abm'), ode, rtol=0.1)
    assert np.isclose(run(dict(home=0.03, school=0.0), 'ode'), run(0.03, 'ode', layers=['home'])) < 0.9*ode # No transmission at school is home contacts only
    for bad in [dict(home=0.03, school=0.02, wrok=0.1), dict(home=0.03)]: # A key with no route, and a route with no key
        with pytest.raises(ss.RouteMismatch):
            run(bad, 'abm')
    return ode


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
    layers = test_layers()

    T.toc()

    if do_plot:
        plt.show()
