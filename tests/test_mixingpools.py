"""
Test mixing pools and contact layers
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss
import starsim.library as ssl
import pytest

n_agents = 1_000
do_plot = False
sc.options(interactive=False) # Assume not running interactively


def run(networks, sir=None, do_plot=False):
    """ Run an SIR over the given mixing pools, and check there were infections """
    sim = ss.Sim(n_agents=n_agents, diseases=sir or ss.SIR(), networks=networks)
    sim.run()
    if do_plot:
        sim.plot()
    cum = sim.results.sir.infection.cum
    assert cum[-1] > cum[0], 'There were no new infections in the simulation'
    return sim


@sc.timer()
def test_single(do_plot=do_plot):
    """ Test a single MixingPool: with defaults, by UIDs, by age, and by sex """
    sc.heading('Testing single mixing pools...')
    sims = sc.objdict()
    sims.defaults = run(ss.MixingPool(), do_plot=do_plot)

    # In this arrangement, src transmits to dst, but can never go the other way, so the epidemic dies out once everyone in src recovers
    k = n_agents // 2
    sims.uids = run(ss.MixingPool(src=ss.uids(np.arange(k)), dst=ss.uids(np.arange(k, n_agents))), do_plot=do_plot)

    # Incidence must decline because 0-15 --> 15+ transmission only
    mp = ss.MixingPool(src=ss.AgeGroup(0, 15), dst=ss.AgeGroup(15, None), beta=1.0, n_contacts=ss.poisson(lam=5))
    sims.age = run(mp, ss.SIR(init_prev=0.1), do_plot=do_plot) # Enough seeds under 15 that some infect

    # Female to male (only) transmission
    mp = ss.MixingPool(src=lambda sim: sim.people.female, dst=lambda sim: sim.people.male, beta=1.0, n_contacts=ss.poisson(lam=4))
    sir = ss.SIR(init_prev=ss.bernoulli(p=lambda self, sim, uids: 0.05*sim.people.female)) # Seed 5% of the female population
    sims.sex = sim = run(mp, sir, do_plot=do_plot)
    assert sim.people.male[sim.diseases.sir.ti_I>0].all() # All new infections should be in men
    return sims


@sc.timer()
def test_single_errors():
    """ Test a single MixingPool with an NCD, and with a missing disease """
    sc.heading('Testing mixing pool errors...')
    mp_pars = dict(src=ss.AgeGroup(0, 15), dst=ss.AgeGroup(15, None), beta=1.0, n_contacts=ss.poisson(lam=5))
    for disease, pool_disease in [(ssl.NCD(), 'ncd'), (ss.SIR(), 'hiv')]:
        sim = ss.Sim(n_agents=n_agents, diseases=disease, networks=ss.MixingPool(diseases=pool_disease, **mp_pars))
        with pytest.raises(Exception):
            sim.run()
    return sim


@sc.timer()
def test_multi(do_plot=do_plot):
    """ Test MixingPools, with defaults and by sex """
    sc.heading('Testing multiple mixing pools...')
    sim1 = run(ss.MixingPools(src={'all':None}, dst={'all':None}, n_contacts=[[1]]), do_plot=do_plot)
    groups = {
        'Female': lambda sim: sim.people.female,
        'Male': lambda sim: sim.people.male,
    }
    mps = ss.MixingPools(n_contacts=np.array([[1.4, 0.5], [1.2, 0.7]]), beta=1.0, src=groups, dst=groups)
    sim2 = run(mps, do_plot=do_plot)
    return sim1, sim2


@sc.timer()
def test_coded_declared():
    sc.heading('Testing that a disease written as code gives the same epidemic on a mixing pool as a declared one')
    S, I, R = ss.states('S I R')
    pars = dict(networks=ss.MixingPool(n_contacts=1), n_agents=5000, stop=ss.days(150), verbose=0)
    coded = ss.Sim(ssl.AgentSIR(beta=ss.perday(0.4), dur_inf=ss.expon(scale=ss.days(7)), p_death=0, init_prev=0.01, name='sir'), **pars).run()
    declared = ss.Sim(ss.Disease(S >> I @ 0.4*I, I >> R @ ss.days(7), init_prev=0.01, name='sir'), **pars).run()
    sizes = coded.results.sir.recovered[-1], declared.results.sir.R[-1]
    assert np.isclose(*sizes, rtol=0.03), sizes # The hazard is multiplied by the share infectious before it is a probability (when it was not, the coded epidemic was about 5% smaller, and its peak a fifth lower)
    return sizes


@sc.timer()
def test_layers():
    sc.heading('Testing contact layers with a beta for each, under both engines')
    def run(beta, solver, layers=('home', 'school')):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
        nets = [ss.contacts('Kenya', layer=layer) for layer in layers]
        sim = ss.Sim(sir, networks=nets, stratify=dict(age=[0, 5, 18, 65]), unit='day', stop=100, solver=solver, total_pop=10_000, verbose=0, **(dict(n_agents=10_000) if solver == 'abm' else {}))
        return sim.run().results.sir.infection.cum[-1]
    ode = run(dict(home=0.03, school=0.02), 'ode') # The spec's layered example, which crashed under both engines
    assert np.isclose(run(dict(home=0.03, school=0.02), 'abm'), ode, rtol=0.1)
    assert np.isclose(run(dict(home=0.03, school=0.0), 'ode'), run(0.03, 'ode', layers=['home'])) < 0.9*ode # No transmission at school is home contacts only
    for bad in [dict(home=0.03, school=0.02, wrok=0.1), dict(home=0.03)]: # A key with no route, and a route with no key
        with pytest.raises(ValueError, match='is given for each route'):
            run(bad, 'abm')
    return ode


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    sims = test_single(do_plot)
    sim = test_single_errors()
    sims2 = test_multi(do_plot)
    layers = test_layers()

    T.toc()

    if do_plot:
        plt.show()
