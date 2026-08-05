"""
Example SIR model in LASER, from laser_core/docs/example.rst

Requires laser_core~=0.5.1
"""
import numpy as np
import matplotlib.pyplot as plt
from laser_core import LaserFrame
from laser_core import PropertySet
import sciris as sc

class SIRModel:
    def __init__(self, params):
        # Model Parameters
        self.params = params

        # Initialize the population LaserFrame
        self.population = LaserFrame(capacity=params.population_size,initial_count=params.population_size)

        # Add disease state property (0 = Susceptible, 1 = Infected, 2 = Recovered)
        self.population.add_scalar_property("disease_state", dtype=np.int32, default=0)

        # Add a recovery timer property (for intrahost progression, optional for timing)
        self.population.add_scalar_property("recovery_timer", dtype=np.int32, default=0)

        # Results tracking
        self.results = LaserFrame( capacity = 1 ) # number of nodes
        self.results.add_vector_property( "S", length=params["timesteps"], dtype=np.float32 )
        self.results.add_vector_property( "I", length=params["timesteps"], dtype=np.float32 )
        self.results.add_vector_property( "R", length=params["timesteps"], dtype=np.float32 )

        # Components
        self.components = []

    def add_component(self, component):
        self.components.append(component)

    def track_results(self, tick):
        susceptible = (self.population.disease_state == 0).sum()
        infected = (self.population.disease_state == 1).sum()
        recovered = (self.population.disease_state == 2).sum()
        total = self.population.count
        self.results.S[tick] = susceptible / total
        self.results.I[tick] = infected / total
        self.results.R[tick] = recovered / total

    def run(self):
        for tick in range(self.params.timesteps):
            for component in self.components:
                component.step()
            self.track_results(tick)

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


class IntrahostProgression:
    def __init__(self, model):
        self.population = model.population

        # Seed the infection
        num_initial_infected = int(0.01 * model.params.population_size)  # e.g., 1% initially infected
        infected_indices = np.random.choice(model.params.population_size, size=num_initial_infected, replace=False)
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


class Transmission:
    def __init__(self, model):
        self.population = model.population
        self.infection_rate = model.params.infection_rate

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


def run_sim(run=True, plot=False):
    """ Create and run the simulation """
    params = PropertySet({
        "population_size": 100_000*2,
        "infection_rate": 0.3,
        "timesteps": 100*10
    })

    # Initialize the model
    sir_model = SIRModel(params)

    # Initialize and add components
    sir_model.add_component(IntrahostProgression(sir_model))
    sir_model.add_component(Transmission(sir_model))

    # Run the simulation
    if run:
        sir_model.run()

    # Plot results
    if plot:
        sir_model.plot_results()
    return sir_model


if __name__ == '__main__':
    with sc.timer():
        sir_model = run_sim()
    sir_model.plot_results()