"""
Run simplest tests
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt

def test_sir():
    ppl = ss.People(10000)
    ppl.networks = ss.ndict(ss.RandomNetwork(n_contacts=ss.poisson(4, rng='Num Contacts')))
    sir = ss.SIR()
    sir.pars['beta'] = {'randomnetwork': 0.1}
    sim = ss.Sim(people=ppl, diseases=sir)
    sim.run()

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
    return


def test_ncd():
    ppl = ss.People(10000)
    ppl.networks = None
    ncd = ss.NCD()
    sim = ss.Sim(people=ppl, diseases=ncd)
    sim.run()

    plt.figure()
    plt.stackplot(
        sim.yearvec,
        ncd.results.n_not_at_risk,
        ncd.results.n_at_risk-ncd.results.n_affected,
        ncd.results.n_affected,
        sim.results.new_deaths.cumsum(),
    )
    plt.legend(['Not at risk','At risk','Affected', 'Dead'])
    plt.xlabel('Year')
    plt.title('NCD')
    return


if __name__ == '__main__':
    sim1 = test_sir()
    #sim2 = test_ncd()
    plt.show()