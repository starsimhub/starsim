"""
Run tests of disease models
"""

# %% Imports and settings
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt

test_run = True
n_agents = [10_000, 2_000][test_run]
do_plot = True
sc.options(interactive=False) # Assume not running interactively


def test_sir():
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
    sir.pars.beta = {'random': 0.1}

    # You can also change the parameters of the default lognormal distribution directly
    sir.pars.dur_inf.set(loc=5)

    # Or use a function, here a lambda that makes the mean for each agent equal to their age divided by 10
    sir.pars.dur_inf.set(loc = lambda self, sim, uids: sim.people.age[uids] / 10)

    sim = ss.Sim(people=ppl, diseases=sir, networks=networks)
    sim.run()

    # CK: parameters changed
    # assert len(sir.log.out_edges(np.nan)) == sir.pars.initial # Log should match initial infections
    df = sir.log.line_list  # Check generation of line-list
    # assert df.source.isna().sum() == sir.pars.initial # Check seed infections in line list

    plt.figure()
    plt.stackplot(
        sim.yearvec,
        sir.results.n_susceptible,
        sir.results.n_infected,
        sir.results.n_recovered,
        sim.results.new_deaths.cumsum(),
    )
    plt.legend(['Susceptible', 'Infected', 'Recovered', 'Dead'])
    plt.xlabel('Year')
    plt.title('SIR')
    return sim


def test_ncd():
    ppl = ss.People(n_agents)
    ncd = ss.NCD()
    sim = ss.Sim(people=ppl, diseases=ncd)
    sim.run()

    assert len(ncd.log.out_edges) == ncd.log.number_of_edges()
    df = ncd.log.line_list  # Check generation of line-list
    assert df.source.isna().all()

    plt.figure()
    plt.stackplot(
        sim.yearvec,
        ncd.results.n_not_at_risk,
        ncd.results.n_at_risk - ncd.results.n_affected,
        ncd.results.n_affected,
        sim.results.new_deaths.cumsum(),
    )
    plt.legend(['Not at risk', 'At risk', 'Affected', 'Dead'])
    plt.xlabel('Year')
    plt.title('NCD')
    return sim


def test_gavi():
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
    ppl = ss.People(n_agents)
    sir1 = ss.SIR(name='sir1')
    sir2 = ss.SIR(name='sir2')

    sir1.pars.beta = {'randomnet': 0.1}
    sir2.pars.beta = {'randomnet': 0.2}
    networks = ss.RandomNet(pars=dict(n_contacts=ss.poisson(4)))

    sim = ss.Sim(people=ppl, diseases=[sir1, sir2], networks=networks)
    sim.run()
    return sim


if __name__ == '__main__':
    sc.options(interactive=do_plot)
    sim1 = test_sir()
    sim2 = test_ncd()
    sims = test_gavi()
    sim = test_multidisease()
