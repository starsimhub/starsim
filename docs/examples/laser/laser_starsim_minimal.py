"""
Minimal port of laser_example_orig.py, following LASER's naming conventions

Requires starsim>=3.0
"""
import numpy as np
import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss

class SIRModel(ss.Module):
    def __init__(self):
        super().__init__()
        self.define_states(
            ss.Arr('disease_state', dtype=np.int32, default=0),
            ss.Arr('recovery_timer', dtype=np.int32, default=0),
        )

    def init_pre(self, sim):
        super().init_pre(sim)
        self.population = self.sim.people
        self.population.disease_state = self.disease_state
        self.population.recovery_timer = self.recovery_timer

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('S', dtype=np.float32),
            ss.Result('I', dtype=np.float32),
            ss.Result('R', dtype=np.float32),
        )

    def step(self):
        tick = self.ti
        susceptible = (self.population.disease_state == 0).sum()
        infected = (self.population.disease_state == 1).sum()
        recovered = (self.population.disease_state == 2).sum()
        total = len(self.population)
        self.results.S[tick] = susceptible / total
        self.results.I[tick] = infected / total
        self.results.R[tick] = recovered / total

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
        model = self.sim
        self.population = model.people

        # Seed the infection
        num_initial_infected = int(0.01 * model.pars.n_agents)  # e.g., 1% initially infected
        infected_indices = np.random.choice(model.pars.n_agents, size=num_initial_infected, replace=False)
        self.population.disease_state[infected_indices] = 1

        # Initialize recovery timer for initially infected individuals
        initially_infected = self.population.disease_state == 1
        self.population.recovery_timer[initially_infected] = np.random.randint(5, 15, size=initially_infected.sum())

    def step(self):
        infected = self.population.disease_state == 1

        # Decrement recovery timer
        self.population.recovery_timer[infected] -= 1

        # Recover individuals whose recovery_timer has reached 0
        recoveries = infected & (self.population.recovery_timer <= 0)
        self.population.disease_state[recoveries] = 2


class Transmission(ss.Module):
    def __init__(self, infection_rate):
        super().__init__()
        self.infection_rate = infection_rate

    def init_post(self):
        super().init_post()
        self.population = self.sim.people

    def step(self):
        susceptible = self.population.disease_state == 0
        infected = self.population.disease_state == 1

        num_susceptible = susceptible.sum()
        num_infected = infected.sum()
        population_size = len(self.population)

        # Fraction of infected and susceptible individuals
        fraction_infected = num_infected / population_size

        # Transmission logic: Probability of infection per susceptible individual
        infection_probability = self.infection_rate * fraction_infected

        # Apply infection probability to all susceptible individuals
        new_infections = np.random.rand(num_susceptible) < infection_probability

        # Set new infections and initialize their recovery_timer
        susceptible_indices = np.where(susceptible)[0]
        newly_infected_indices = susceptible_indices[new_infections]
        self.population.disease_state[newly_infected_indices] = 1
        self.population.recovery_timer[newly_infected_indices] = np.random.randint(5, 15, size=newly_infected_indices.size)  # Random recovery time


params = sc.objdict({
    "n_agents": 100_000,
    "dur": 160,
})
infection_rate = 0.3

intrahost = IntrahostProgression()
transmission = Transmission(infection_rate)
sir_model = SIRModel()

# Initialize the model
sim = ss.Sim(params, modules=[intrahost, transmission, sir_model], copy_inputs=False)

# Run the simulation
with sc.timer():
    sim.run()

# Plot results
sir_model.plot_results()