"""
Test births and deaths
"""

# %% Imports and settings
import starsim as ss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sciris as sc

class agent_analyzer(ss.Analyzer):
    def init_pre(self, sim):
        super().init_pre(sim)
        self.n_agents = np.zeros(sim.npts)

    def update_results(self, sim):
        self.n_agents[sim.ti] = len(sim.people)


def run_test(remove_dead=True, do_plot=False, rand_seed=0, verbose=False):
    ppl = ss.People(50000)

    # Parameters
    realistic_birth = {'birth_rate': pd.read_csv(ss.root/'tests/test_data/nigeria_births.csv')}

    series_death = {
        'death_rate': pd.Series(
            index=[0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
            data=[0.0046355, 0.000776, 0.0014232, 0.0016693, 0.0021449, 0.0028822, 0.0039143, 0.0053676, 0.0082756, 0.01, 0.02, 0.03, 0.04, 0.06, 0.11, 0.15, 0.21, 0.30],
        ),
        'rel_death': 7
    }

    births = ss.Births(realistic_birth)
    deaths = ss.Deaths(series_death)
    gon = ss.Gonorrhea({'p_death': 0.5, 'init_prev': 0.1})

    sim = ss.Sim(people=ppl, demographics=[births,deaths], diseases=[gon], networks=ss.MFNet(), remove_dead=remove_dead, n_years=100, analyzers=[agent_analyzer()], rand_seed=rand_seed, verbose=verbose)

    sim.initialize()
    sim.run()

    if do_plot:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(sim.tivec, sim.results.births.new, label='Births')
        ax[0].plot(sim.tivec, sim.results.background_deaths.new, label='Deaths')
        ax[1].plot(sim.tivec, sim.results.n_alive, label='Number of alive agents')
        ax[1].plot(sim.tivec, sim.analyzers[0].n_agents, label='Number of agents')
        ax[0].set_title('Births and deaths')
        ax[1].set_title('Population size')
        ax[0].legend()
        ax[1].legend()
        fig.suptitle(f'Remove people {remove_dead}')
        fig.tight_layout()
        plt.show()

    return sim


def compare_totals(rand_seed=0):
    sim1 = run_test(False, rand_seed=rand_seed)
    sim2 = run_test(True, rand_seed=rand_seed+1)
    return sim1.people.alive.count_nonzero() - sim2.people.alive.count_nonzero()


if __name__ == '__main__':

    # x = sc.parallelize(compare_totals, np.arange(50))

    sim1 = run_test(False, rand_seed=0, verbose=0.1)
    sim2 = run_test(True, rand_seed=0, verbose=0.1)

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()

    ax[0].plot(sim1.tivec, sim1.results.births.new, label='No removal')
    ax[0].plot(sim2.tivec, sim2.results.births.new, label='Removal', linestyle='--')
    ax[0].set_title('Births')


    ax[1].plot(sim1.tivec, sim1.results.deaths.new, label='No removal')
    ax[1].plot(sim2.tivec, sim2.results.deaths.new, label='Removal', linestyle='--')
    ax[1].set_title('Background deaths')

    ax[1].plot(sim1.tivec, sim1.results.new_deaths, label='(all) No removal')
    ax[1].plot(sim2.tivec, sim2.results.new_deaths, label='(all) Removal', linestyle='--')


    ax[2].plot(sim1.tivec, sim1.results.n_alive, label='No removal')
    ax[2].plot(sim2.tivec, sim2.results.n_alive, label='Removal', linestyle='--')
    ax[2].set_title('Number of alive agents')


    ax[3].plot(sim1.tivec, sim1.analyzers[0].n_agents, label='No removal')
    ax[3].plot(sim2.tivec, sim2.analyzers[0].n_agents, label='Removal', linestyle='--')
    ax[3].set_title('Number of agents')

    fig.tight_layout()
    plt.show()
