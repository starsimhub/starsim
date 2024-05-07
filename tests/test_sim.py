"""
Test simple APIs
"""

# %% Imports and settings
import starsim as ss
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt

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
    assert not (ss.diff_sims(s1, s2) or ss.diff_sims(s2, s3)), 'Sims should match'
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
    sim3b, sim3i = test_simple_vax(do_plot=do_plot)
    sim4 = test_components(do_plot=do_plot)
    sim5a, sim5b = test_parallel()
    
    T.toc()
    
    if do_plot:
        plt.show()
