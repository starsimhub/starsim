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
    sim = ss.demo(plot=do_plot)
    return sim


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


def test_sir_epi():
    sc.heading('Test basic epi dynamics')

    # Define the parameters to vary
    par_effects = dict(
        beta=[0.01, 0.99],
        n_contacts=[1, 20],
        init_prev=[0.1, 0.9],
        dur_inf=[1, 8],
        p_death=[.01, .1],
    )

    # Loop over each of the above parameters and make sure they affect the epi dynamics in the expected ways
    for par, par_val in par_effects.items():
        lo = par_val[0]
        hi = par_val[1]

        # Make baseline pars
        pars0 = make_sim_pars()
        pars1 = make_sim_pars()

        if par != 'n_contacts':
            pars0['diseases'] = sc.mergedicts(pars0['diseases'], {par: lo})
            pars1['diseases'] = sc.mergedicts(pars1['diseases'], {par: hi})
        else:
            pars0['networks'] = sc.mergedicts(pars0['networks'], {par: lo})
            pars1['networks'] = sc.mergedicts(pars1['networks'], {par: hi})

        # Run the simulations and pull out the results
        s0 = ss.Sim(pars0, label=f'{par} {par_val[0]}').run()
        s1 = ss.Sim(pars1, label=f'{par} {par_val[1]}').run()

        # Check results
        if par == 'p_death':
            v0 = s0.results.cum_deaths[-1]
            v1 = s1.results.cum_deaths[-1]
        else:
            ind = 1 if par == 'init_prev' else -1
            v0 = s0.results.sir.cum_infections[ind]
            v1 = s1.results.sir.cum_infections[ind]

        print(f'Checking with varying {par:10s} ... ', end='')
        assert v0 <= v1, f'Expected infections to be lower with {par}={lo} than with {par}={hi}, but {v0} > {v1})'
        print(f'✓ ({v0} <= {v1})')

    return s0, s1


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


def test_sis(do_plot=do_plot):
    pars = dict(
        n_agents = n_agents,
        diseases = 'sis',
        networks = 'random'
    )
    sim = ss.Sim(pars)
    sim.run()
    if do_plot:
        sim.plot()
        sim.diseases.sis.plot()
    return sim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()
    
    s0 = test_demo(do_plot=do_plot)
    s1 = test_default(do_plot=do_plot)
    s2 = test_simple(do_plot=do_plot)
    s3a, s3b = test_sir_epi()
    s4_base, s4_intv = test_simple_vax(do_plot=do_plot)
    s5 = test_components(do_plot=do_plot)
    s6a, s6b = test_parallel()
    s7 = test_sis(do_plot=do_plot)
    
    T.toc()
    
    if do_plot:
        plt.show()
