import sciris as sc
import pylab as pl
import starsim as ss
import pandas as pd


# %% Define the tests
def test_dx():
    ppl = ss.People(1000)
    ppl.networks = ss.RandomNetwork(n_contacts=2)

    sir = ss.SIR()
    sir.pars['beta'] = {'RandomNetwork': 0.08}

    dxdf = pd.DataFrame([
        ['susceptible', 'positive', 0.01],
        ['susceptible', 'negative', 0.99],
        ['infected', 'positive', 0.95],
        ['infected', 'negative', 0.05],
        ['recovered', 'positive', 0.30],
        ['recovered', 'negative', 0.70],
    ], columns=['state', 'result', 'probability'])
    dxdf['disease'] = 'sir'

    dx = ss.dx(dxdf)
    test = ss.campaign_triage(product=dx, prob=0.6, annual_prob=True, years=2025, eligibility=lambda sim: sim.people.alive)
    
    sim = ss.Sim(start=2024, end=2030, people=ppl, diseases=sir, interventions=test)
    sim.initialize()
    sim.run()

    return sim


# %% Run as a script
if __name__ == '__main__':
    # Start timing
    T = sc.tic()

    # Run tests
    sim = test_dx()
    sc.toc(T)

    # Plot
    pl.plot(sim.tivec, sim.results.sir.n_infected) # n_susceptible, n_infected, n_recovered, prevalence, new_infections
    pl.show()
    print('Done.')