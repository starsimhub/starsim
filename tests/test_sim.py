"""
Test simple APIs
"""

# %% Imports and settings
import starsim as ss
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
import pytest

n_agents = 1_000
do_plot = False
sc.options(interactive=False) # Assume not running interactively


def make_sim_pars():
    pars = sc.objdict(
        n_agents = n_agents,
        birth_rate = 20,
        death_rate = 15,
        networks = sc.objdict(
            type = 'randomnet',
            n_contacts = 4,
        ),
        diseases = sc.objdict(
            type = 'sir',
            dur_inf = 10,
            beta = 0.1,
        )
    )
    return pars


def test_demo(do_plot=do_plot):
    """ Test Starsim's demo run """
    s1 = ss.demo(plot=do_plot)
    
    # Test explicit demo
    s2 = ss.Sim(diseases='sir', networks='random').run()
    
    # Test explicit 
    sir = ss.SIR()
    net = ss.RandomNet()
    s3 = ss.Sim(diseases=sir, networks=net).run()
    assert ss.check_sims_match(s1, s2, s3), 'Sims should match'
    return s1


def test_default(do_plot=do_plot):
    """ Create, run, and plot a sim with default settings """
    sim = ss.Sim(n_agents=n_agents).run()
    if do_plot:
        sim.plot()
    return sim


def test_simple(do_plot=do_plot):
    """ Create, run, and plot a sim by passing a parameters dictionary """
    pars = make_sim_pars()
    sim = ss.Sim(pars)
    sim.run()
    if do_plot:
        sim.plot()
    return sim


def test_api():
    """ Test all different ways of creating a sim """
    
    # Check different ways of specifying a sim
    s1 = ss.Sim(n_agents=n_agents, diseases='sir', networks='random').run() # Supply strings directly
    s2 = ss.Sim(pars=dict(n_agents=n_agents, diseases='sir', networks='random')).run() # Supply as parameters
    s3 = ss.Sim(n_agents=n_agents, diseases=ss.SIR(), networks=ss.RandomNet()).run() # Supply as objects
    ss.check_sims_match(s1, s2, s3), 'Sims should match'
    
    # Check different ways of setting a distribution
    kw = dict(n_agents=n_agents, networks='random')
    d1 = ss.lognorm_ex(10) # Create a distribution with an argument
    d2 = ss.lognorm_ex(mean=10, stdev=2) # Create a distribution with kwargs
    d3 = ss.normal(loc=10) # Create a different type of distribution
    
    # Check specifying dist with a scalar
    s4 = ss.Sim(diseases=dict(type='sir', dur_inf=10), **kw).run() # Supply values as a scalar
    s5 = ss.Sim(diseases=dict(type='sir', dur_inf=d1), **kw).run() # Supply as a distribution
    ss.check_sims_match(s4, s5), 'Sims should match'
    
    # Check specifying dist with a list and dict
    s6 = ss.Sim(diseases=dict(type='sir', dur_inf=[10,2]), **kw).run() # Supply values as a list
    s7 = ss.Sim(diseases=dict(type='sir', dur_inf=dict(mean=10, stdev=2)), **kw).run() # Supply values as a dict
    s8 = ss.Sim(diseases=dict(type='sir', dur_inf=d2), **kw).run() # Supply as a distribution
    ss.check_sims_match(s6, s7, s8), 'Sims should match'
    
    # Check changing dist type
    s9  = ss.Sim(diseases=dict(type='sir', dur_inf=dict(type='normal', loc=10)), **kw).run() # Supply values as a dict
    s10 = ss.Sim(diseases=dict(type='sir', dur_inf=d3), **kw).run() # Supply values as a distribution
    ss.check_sims_match(s9, s10), 'Sims should match'
    
    # # Check that Bernoulli distributions can't be changed
    # with pytest.raises(TypeError):
    #     ss.Sim(diseases=dict(type='sir', init_prev=dict(type='normal', loc=10)), **kw).initialize()
    
    return s1


def test_simple_vax(do_plot=do_plot):
    """ Create and run a sim with vaccination """
    ss.set_seed(1)
    pars = make_sim_pars()
    sim_base = ss.Sim(pars=pars)
    sim_base.run()

    my_vax = ss.sir_vaccine(pars=dict(efficacy=0.5))
    intv = ss.routine_vx(start_year=2015, prob=0.2, product=my_vax)
    sim_intv = ss.Sim(pars=pars, interventions=intv)
    sim_intv.run()
    
    assert sim_intv.summary.cum_deaths < sim_base.summary.cum_deaths, 'Vaccine should avert deaths'

    # Check plots
    if do_plot:
        pi = 0

        plt.figure()
        plt.plot(sim_base.yearvec[pi:], sim_base.results.sir.prevalence[pi:], label='Baseline')
        plt.plot(sim_intv.yearvec[pi:], sim_intv.results.sir.prevalence[pi:], label='Vax')
        plt.axvline(x=2015, color='k', ls='--')
        plt.title('Prevalence')
        plt.legend()

    return sim_base, sim_intv


def test_components(do_plot=do_plot):
    """ Create, run, and plot a sim by assembling components """
    people = ss.People(n_agents=n_agents)
    network = ss.RandomNet(pars=dict(n_contacts=4))
    sir = ss.SIR(pars=dict(dur_inf=10, beta=0.1))
    sim = ss.Sim(diseases=sir, people=people, networks=network)
    sim.run()
    if do_plot:
        sim.plot()
    return sim


def test_parallel():
    """ Test running two identical sims in parallel """
    pars = make_sim_pars()

    # Check that two identical sims match
    sims = ss.MultiSim([ss.Sim(pars, label='Sim1'), ss.Sim(pars, label='Sim2')])
    sims.run(keep_people=True)
    s1, s2 = sims.sims
    assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)

    # Check that two non-identical sims don't match
    pars2 = sc.dcp(pars)
    pars2.diseases.beta *= 2
    sims = ss.MultiSim([ss.Sim(pars, label='Sim1'), ss.Sim(pars2, label='Sim2')])
    sims.run(keep_people=True)
    s1, s2 = sims.sims
    assert not np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)

    return s1, s2


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()
    
    sim0 = test_demo(do_plot=do_plot)
    sim1 = test_default(do_plot=do_plot)
    sim2 = test_simple(do_plot=do_plot)
    sim3 = test_api()
    sim4b, sim4i = test_simple_vax(do_plot=do_plot)
    sim5 = test_components(do_plot=do_plot)
    sim6a, sim6b = test_parallel()
    
    T.toc()
    
    if do_plot:
        plt.show()
