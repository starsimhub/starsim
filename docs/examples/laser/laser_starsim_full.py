"""
Fuller port of laser_example_orig.py, using Starsim's naming conventions

Requires starsim>=3.0
"""
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
import starsim as ss


class SIRModel(ss.Module):
    def __init__(self):
        super().__init__()
        self.define_states(
            ss.Arr('disease_state', dtype=np.int32, default=0),
            ss.Arr('recovery_timer', dtype=np.int32, default=0),
        )

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('S'),
            ss.Result('I'),
            ss.Result('R'),
        )

    def step(self):
        ti = self.ti
        susceptible = (self.disease_state == 0).sum()
        infected = (self.disease_state == 1).sum()
        recovered = (self.disease_state == 2).sum()
        total = len(self.sim.people)
        self.results.S[ti] = susceptible / total
        self.results.I[ti] = infected / total
        self.results.R[ti] = recovered / total

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.results.S, label="Susceptible (S)", color="blue")
        plt.plot(self.results.I, label="Infected (I)", color="red")
        plt.plot(self.results.R, label="Recovered (R)", color="green")
        plt.title("SIR Model Dynamics with LASER Components")
        plt.xlabel("Time (Timesteps)")
        plt.ylabel("Fraction of Population")
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig("gpt_sir.png")


class IntrahostProgression(ss.Module):
    def init_post(self):
        super().init_post()
        n_agents = self.sim.pars.n_agents
        sir = self.sim.people.sirmodel

        # Seed the infection
        num_initial_infected = int(0.01 * n_agents)  # e.g., 1% initially infected
        infected_indices = np.random.choice(n_agents, size=num_initial_infected, replace=False)
        sir.disease_state[infected_indices] = 1

        # Initialize recovery timer for initially infected individuals
        initially_infected = sir.disease_state == 1
        sir.recovery_timer[initially_infected] = np.random.randint(5, 15, size=initially_infected.sum())

    def step(self):
        sir = self.sim.people.sirmodel

        infected = sir.disease_state == 1

        # Decrement recovery timer
        sir.recovery_timer[infected] -= 1

        # Recover individuals whose recovery_timer has reached 0
        recoveries = infected & (sir.recovery_timer <= 0)
        sir.disease_state[recoveries] = 2


class Transmission(ss.Module):
    def __init__(self, infection_rate):
        super().__init__()
        self.infection_rate = infection_rate

    def step(self):
        sir = self.sim.people.sirmodel
        susceptible = sir.disease_state == 0
        infected = sir.disease_state == 1

        num_susceptible = susceptible.sum()
        num_infected = infected.sum()
        population_size = len(self.sim.people)

        # Fraction of infected and susceptible individuals
        fraction_infected = num_infected / population_size

        # Transmission logic: Probability of infection per susceptible individual
        infection_probability = self.infection_rate * fraction_infected

        # Apply infection probability to all susceptible individuals
        new_infections = np.random.rand(num_susceptible) < infection_probability

        # Set new infections and initialize their recovery_timer
        susceptible_indices = np.where(susceptible)[0]
        newly_infected_indices = susceptible_indices[new_infections]
        sir.disease_state[newly_infected_indices] = 1
        sir.recovery_timer[newly_infected_indices] = np.random.randint(5, 15, size=newly_infected_indices.size)  # Random recovery time


def run_sim(run=True, plot=False):
    """ Create and run the simulation """
    pars = dict(
        n_agents = 100_000*2,
        dur = 100*10,
        verbose = 0,
    )
    infection_rate = 0.3

    intrahost = IntrahostProgression()
    transmission = Transmission(infection_rate)
    sir_model = SIRModel()

    # Initialize the model
    sim = ss.Sim(pars, modules=[intrahost, transmission, sir_model], copy_inputs=False)

    # Run the simulation
    if run:
        sim.run()

    # Plot results
    if plot:
        sir_model.plot_results()
    return sim


if __name__ == '__main__':
    with sc.timer():
        sir_model = run_sim()
    sir_model.plot_results()