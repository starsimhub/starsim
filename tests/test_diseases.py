"""
Run tests of disease models
"""

# %% Imports and settings
import starsim as ss
import matplotlib.pyplot as plt
import scipy.stats as sps
import numpy as np

test_run = True
n_agents = [10_000, 2_000][test_run]


def test_sir():
    ppl = ss.People(n_agents)

    sir_pars = {
        'dur_inf': sps.norm(loc=10),  # Override the default distribution
    }
    sir = ss.SIR(sir_pars)

    # You can also change the parameters of the default lognormal distribution directly!
    # sir.pars['dur_inf'].kwds['loc'] = 5

    # Or why not put a lambda here for fun!
    sir.pars['dur_inf'].kwds['loc'] = lambda self, sim, uids: sim.people.age[uids] / 10

    sir.pars['beta'] = {'random': 0.1}
    networks = ss.random(pars=dict(n_contacts=sps.poisson(mu=4)))

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


# @pytest.mark.skip(reason="Haven't converted yet")
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


def test_gavi(disease):
    ss.options(multirng=False)
    pars = dict(
        diseases=disease,
        n_agents=n_agents,
        networks='random',
    )
    sim = ss.Sim(pars)
    sim.run()
    return sim


if __name__ == '__main__':
    ss.options(multirng=False)
    sim1 = test_sir()
    sim2 = test_ncd()
    for disease in ['cholera', 'measles']:
        sim = test_gavi(disease)
    plt.show()
