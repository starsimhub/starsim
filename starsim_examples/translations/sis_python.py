"""
Test simulation performance
"""
import numpy as np
import sciris as sc

# Default dtype
npfloat = np.float32

def floatzeros(*args, **kwargs):
    """ Create an array with the default dtype """
    return np.zeros(*args, dtype=npfloat, **kwargs)

def boolzeros(*args, **kwargs):
    """ Create a boolean array """
    return np.zeros(*args, dtype=bool, **kwargs)

def lognormal_ex(mean=0, std=1, size=1):
    """ Create a lognormal distribution with the specified mean and exponential SD """
    # Calculate the parameters of the underlying normal distribution
    sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
    mu = np.log(mean) - (sigma ** 2) / 2
    rvs = np.random.lognormal(mean=mu, sigma=sigma, size=size)
    return rvs

# Define the parameters
pars = sc.dictobj(
    n_agents = 100_000,
    dur = 100,
)

class Sim(sc.prettyobj):
    """ Minimal translation of Starsim's Sim class"""
    pass


class Random(sc.prettyobj):
    """ Minimal translation of Starsim's Random network """
    pass


class SIS(sc.prettyobj):
    """ Minimal translation of Starsim's SIS model """

    def __init__(self, pars, beta=0.05, init_prev=0.01, dur_inf=10, waning=0.05, imm_boost=1.0):

        # Set parameters
        self.n_agents = pars.n_agents
        self.dur = pars.dur
        self.beta = beta
        self.init_prev = init_prev
        self.dur_inf = dur_inf
        self.waning = waning
        self.imm_boost = imm_boost
        self.ti = np.nan

        # Create states
        self.susceptible = boolzeros(self.n_agents)
        self.infected = boolzeros(self.n_agents)
        self.ti_recovered = floatzeros(self.n_agents)
        self.immunity = floatzeros(self.n_agents)
        self.rel_sus = floatzeros(self.n_agents)

        # Create results
        self.n_susceptible = floatzeros(self.dur)
        self.n_infected = floatzeros(self.dur)
        self.mean_rel_sus = floatzeros(self.dur)
        return
    
    def step(self):
        """ Progress the simulation by one time step """
        self.step_state()
        self.update_immunity()
        self.update_results()
        self.ti += 1
        return

    def step_state(self):
        """ Progress infectious -> recovered """
        recovered = (self.infected & (self.ti_recovered <= self.ti))
        self.infected[recovered] = False
        self.susceptible[recovered] = True
        self.update_immunity()        
        return

    def update_immunity(self):
        has_imm = (self.immunity > 0)
        self.immunity[has_imm] *= (1-self.waning)
        self.rel_sus[has_imm] = np.maximum(0, 1 - self.immunity[has_imm])
        self.mean_rel_sus[self.ti] = self.rel_sus.mean()
        return
    
    def update_results(self):
        self.n_susceptible[self.ti] = self.susceptible.sum()
        self.n_infected[self.ti] = self.infected.sum()
        return

    def set_prognoses(self, uids, sources=None):
        """ Set prognoses """
        super().set_prognoses(uids, sources)
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = self.ti
        self.immunity[uids] += self.imm_boost

        # Sample duration of infection
        dur_inf = lognormal_ex(mean=self.dur_inf, size=len(uids))

        # Determine when people recover
        self.ti_recovered[uids] = self.ti + dur_inf

        return

# Create the sim
sim = Sim(
    n_agents=pars.n_agents,
    dur=pars.dur,
    diseases = SIS(pars),
    networks = Random(),
    verbose = 0,
)

# Run and time
T = sc.timer()
sim.run()
T.toc(f'Time for SIS-Python, n_agents={pars.n_agents}, dur={pars.dur}')