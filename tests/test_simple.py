"""
Test simple APIs
"""

# %% Imports and settings
import starsim as ss
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt

n_agents = 1_000
do_plot = True
sc.options(interactive=False) # Assume not running interactively


def test_demo():
    """ Test Starsim's demo run """
    sim = ss.demo()
    return sim


def test_default():
    """ Create, run, and plot a sim with default settings """
    sim = ss.Sim(n_agents=n_agents).run()
    sim.plot()
    return sim


def make_sim(par=None, val=None, label=None):
    # Build a simulation
    n_contacts = 4
    sir_pars = dict(beta=0.1, dur_inf=ss.lognorm_o(mean=10, stdev=1))

    if par == 'n_contacts':
        n_contact = val
    else:
        sir_pars = ss.omergeleft(sir_pars, dict(par=val))

    label = f'{par} {val}' if par and val and (not label) else 'None'

    pop = ss.People(n_agents=n_agents) 
    demog = [ss.Births(birth_rate=20), ss.Deaths(death_rate=15)]
    nets = ss.RandomNet(n_contacts=n_contacts)
    sir = ss.SIR(pars=sir_pars)

    sim = ss.Sim(people=pop, demographics=demog, networks=nets, diseases=sir, label=label)
    return sim


def test_simple():
    """ Create, run, and plot a sim by passing a parameters dictionary """
    sim = make_sim()
    sim.run()
    sim.plot()
    return sim


def test_sir_epi():
    sc.heading('Test basic epi dynamics')

    # Define the parameters to vary
    par_effects = dict(
        init_prev = [0.1, 0.9],
        beta = [0.01, 0.99],
        n_contacts = [1, 20],
        dur_inf = [ss.lognorm_o(mean=1, stdev=1), ss.lognorm_o(mean=8, stdev=1)],
        p_death = [.01, .1],
    )

    # Loop over each of the above parameters and make sure they affect the epi dynamics in the expected ways
    for par, par_val in par_effects.items():
        lo = par_val[0]
        hi = par_val[1]

        # Make baseline pars
        s0 = make_sim({par:par_val[0]})
        s1 = make_sim({par:par_val[1]})

        # Run the simulations and pull out the results
        s0.run()
        s1.run()

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


def test_simple_vax(do_plot=False):
    """ Create and run a sim with vaccination """
    ss.set_seed(1)
    sim_base = make_sim()
    sim_base.run()

    my_vax = ss.sir_vaccine(pars=dict(efficacy=0.5))
    intv = ss.routine_vx(start_year=2015, prob=0.2, product=my_vax)
    sim_intv = make_sim()
    sim_intv.interventions += [intv]
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


def test_components():
    """ Create, run, and plot a sim by assembling components """
    people = ss.People(n_agents=n_agents)
    network = ss.RandomNet(pars=dict(n_contacts=4))
    sir = ss.SIR(pars=dict(beta=0.1))
    sir.pars.dur_inf['mean'] = 10
    sim = ss.Sim(diseases=sir, people=people, networks=network)
    sim.run()
    sim.plot()
    return sim


def test_parallel():
    """ Test running two identical sims in parallel """
    sims = ss.MultiSim([make_sim(label='Sim1'), make_sim(label='Sim2')])
    sims.run(keep_people=True)
    s1, s2 = sims.sims
    assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)
    return s1, s2


if __name__ == '__main__':
    sc.options(interactive=do_plot)
    T = sc.timer()
    
    s0 = test_demo()
    s1 = test_default()
    s2 = test_simple()
    s3a, s3b = test_sir_epi()
    s4_base, s4_intv = test_simple_vax(do_plot=True)
    s5 = test_components()
    s6a, s6b = test_parallel()
    
    T.toc()

    plt.show()
