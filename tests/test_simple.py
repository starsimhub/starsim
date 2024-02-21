"""
Test simple APIs
"""

# %% Imports and settings
import starsim as ss
import numpy as np
import sciris as sc

n_agents = 2_000


def test_default():
    """ Create, run, and plot a sim with default settings """
    sim = ss.Sim(n_agents=n_agents).run()
    sim.plot()
    return sim


def make_sim_pars():
    pars = dict(
        n_agents = n_agents,
        birth_rate = 20,
        death_rate = 0.015,
        networks = dict(
            type = 'randomnet',
            n_contacts = 4,
        ),
        diseases = dict(
            type = 'sir',
            dur_inf = 10,
            beta = 0.1,
        )
    )
    return pars


def test_simple():
    """ Create, run, and plot a sim by passing a parameters dictionary """
    pars = make_sim_pars()
    sim = ss.Sim(pars)
    sim.run()
    sim.plot()
    return sim


def test_sir_epi():
    sc.heading('Test basic epi dynamics')

    # Define the parameters to vary
    par_effects = dict(
        beta=[0.01, 0.99],
        n_contacts=[1, 20],
        init_prev=[0.1, 0.8],
        dur_inf=[1, 8],
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
        v0 = s0.results.sir.cum_infections[-1]
        v1 = s1.results.sir.cum_infections[-1]
        print(f'Checking with varying {par:10s} ... ', end='')
        assert v0 <= v1, f'Expected infections to be lower with {par}={lo} than with {par}={hi}, but {v0} > {v1})'
        print(f'✓ ({v0} <= {v1})')

    return s0, s1



def test_simple_vax(do_plot=False):
    """ Create and run a sim with vaccination """
    ss.set_seed(1)
    pars = make_sim_pars()
    sim_base = ss.Sim(pars=pars)
    sim_base.run()

    my_vax = ss.sir_vaccine(pars=dict(efficacy=0.5))
    sim_intv = ss.Sim(pars=pars, interventions=ss.routine_vx(start_year=2015, prob=0.2, product=my_vax))
    sim_intv.run()

    # Check plots
    if do_plot:
        import matplotlib.pyplot as plt
        pi = 0

        plt.figure()
        plt.plot(sim_base.yearvec[pi:], sim_base.results.sir.prevalence[pi:], label='Baseline')
        plt.plot(sim_intv.yearvec[pi:], sim_intv.results.sir.prevalence[pi:], label='Vax')
        plt.axvline(x=2015, color='k', ls='--')
        plt.title('Prevalence')
        plt.legend()
        plt.show()

    return sim_base, sim_intv


def test_components():
    """ Create, run, and plot a sim by assembling components """
    people = ss.People(n_agents=n_agents)
    network = ss.RandomNet(pars=dict(n_contacts=4))
    sir = ss.SIR(pars=dict(dur_inf=10, beta=0.1))
    sim = ss.Sim(diseases=sir, people=people, networks=network)
    sim.run()
    sim.plot()
    return sim


def test_parallel():
    """ Test running two identical sims in parallel """
    pars = make_sim_pars()
    s1 = ss.Sim(pars)
    s2 = ss.Sim(pars)
    # s1, s2 = ss.parallel(s1, s2).sims
    # assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)
    return s1, s2


if __name__ == '__main__':
    T = sc.timer()
    
    s1 = test_default()
    s2 = test_simple()
    s3a, s3b = test_sir_epi()
    s4_base, s4_intv = test_simple_vax(do_plot=True)
    s5 = test_components()
    s6a, s6b = test_parallel()
    
    T.toc()
