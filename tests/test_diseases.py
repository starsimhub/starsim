"""
Run tests of disease models
"""

# %% Imports and settings
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt

test_run = True
n_agents = [10_000, 2_000][test_run]
do_plot = False
sc.options(interactive=do_plot) # Assume not running interactively


def test_sir():
    sc.heading('Testing SIR dynamics')

    ppl = ss.People(n_agents)
    network_pars = {
        'n_contacts': ss.poisson(4), # Contacts Poisson distributed with a mean of 4
    }
    networks = ss.RandomNet(pars=network_pars)

    sir_pars = {
        'dur_inf': ss.normal(loc=10),  # Override the default distribution
    }
    sir = ss.SIR(sir_pars)

    # Change pars after creating the SIR instance
    sir.pars.beta = {'random': ss.rate(0.1)}

    # You can also change the parameters of the default lognormal distribution directly
    sir.pars.dur_inf.set(loc=5)

    # Or use a function, here a lambda that makes the mean for each agent equal to their age divided by 10
    sir.pars.dur_inf.set(loc = lambda self, sim, uids: sim.people.age[uids] / 10)

    sim = ss.Sim(people=ppl, diseases=sir, networks=networks)
    sim.run()

    plt.figure()
    res = sim.results
    plt.stackplot(
        sim.timevec,
        res.sir.n_susceptible,
        res.sir.n_infected,
        res.sir.n_recovered,
        res.new_deaths.cumsum(),
    )
    plt.legend(['Susceptible', 'Infected', 'Recovered', 'Dead'])
    plt.xlabel('Year')
    plt.title('SIR')
    return sim


def test_sir_epi():
    sc.heading('Test basic epi dynamics')

    base_pars = dict(n_agents=n_agents, networks=dict(type='random'), diseases=dict(type='sir'))

    # Define the parameters to vary
    par_effects = dict(
        beta = [0.01, 0.99],
        n_contacts = [1, 20],
        init_prev = [0.1, 0.9],
        dur_inf = [1, 8],
        p_death = [.01, .1],
    )

    # Loop over each of the above parameters and make sure they affect the epi dynamics in the expected ways
    for par, par_val in par_effects.items():
        lo = par_val[0]
        hi = par_val[1]

        # Make baseline pars
        pars0 = sc.dcp(base_pars)
        pars1 = sc.dcp(base_pars)

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


def test_sis(do_plot=do_plot):
    sc.heading('Testing SIS dynamics')

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


def test_ncd():
    sc.heading('Testing NCDs')

    ppl = ss.People(n_agents)
    ncd = ss.NCD(pars={'log':True})
    sim = ss.Sim(people=ppl, diseases=ncd, copy_inputs=False) # Since using ncd directly below
    sim.run()

    assert len(ncd.log.out_edges) == ncd.log.number_of_edges()
    df = ncd.log.line_list()  # Check generation of line-list
    assert df.source.isna().all()

    plt.figure()
    plt.stackplot(
        sim.timevec,
        ncd.results.n_not_at_risk,
        ncd.results.n_at_risk - ncd.results.n_affected,
        ncd.results.n_affected,
        sim.results.new_deaths.cumsum(),
    )
    assert ncd.results.n_not_at_risk.label == 'Not at risk'
    assert ncd.results.n_affected.label == 'Affected'
    plt.legend(['Not at risk', 'At risk', 'Affected', 'Dead'])
    plt.xlabel('Year')
    plt.title('NCD')
    return sim


def test_gavi():
    sc.heading('Testing GAVI diseases')

    sims = sc.autolist()
    for disease in ['cholera', 'measles', 'ebola']:
        pars = dict(
            diseases = disease,
            n_agents = n_agents,
            networks = 'random',
        )
        sim = ss.Sim(pars)
        sim.run()
        sims += sim
    return sims


def test_multidisease():
    sc.heading('Testing simulating multiple diseases')

    ppl = ss.People(n_agents)
    sir1 = ss.SIR(name='sir1')
    sir2 = ss.SIR(name='sir2')

    sir1.pars.beta = {'randomnet': 0.1}
    sir2.pars.beta = {'randomnet': 0.2}
    networks = ss.RandomNet(pars=dict(n_contacts=ss.poisson(4)))

    sim = ss.Sim(people=ppl, diseases=[sir1, sir2], networks=networks)
    sim.run()
    return sim


def test_mtct():
    sc.heading('Test mother-to-child transmission routes')
    ppl = ss.People(n_agents)
    sis = ss.SIS(beta={'random':[0.005, 0.001], 'prenatal':[0.1, 0], 'postnatal':[0.1, 0]})
    networks = [ss.RandomNet(), ss.PrenatalNet(), ss.PostnatalNet()]
    demographics = ss.Pregnancy(fertility_rate=20)
    sim = ss.Sim(dt=1/12, people=ppl, diseases=sis, networks=networks, demographics=demographics)
    sim.run()
    return sim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    sir   = test_sir()
    s1,s2 = test_sir_epi()
    sis   = test_sis()
    ncd   = test_ncd()
    gavi  = test_gavi()
    multi = test_multidisease()
    mtct  = test_mtct()
