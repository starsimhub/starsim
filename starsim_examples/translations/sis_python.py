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
    def __init__(self, pars, network, disease, verbose=0):
        self.n_agents = pars.n_agents
        self.dur = pars.dur
        self.disease = disease
        self.network = network
        self.verbose = verbose
        self.ti = 0
        return
    
    def step(self):
        net = self.network
        dis = self.disease
        net.step()
        dis.step()
        return

    def run(self):
        for t in range(self.dur):
            self.step()
        return


class RandomNet(sc.prettyobj):
    """ Minimal translation of Starsim's Random network """

    def __init__(self, pars, n_contacts=10, dur=0, beta=1.0):

        # Set parameters
        self.n_agents = pars.n_agents
        self.n_contacts = n_contacts
        self.dur = dur
        self.beta = beta

        # Initialize edge arrays
        self.p1 = np.empty(0, dtype=int)
        self.p2 = np.empty(0, dtype=int)
        self.edge_beta = np.empty(0, dtype=npfloat)
        self.edge_dur = np.empty(0, dtype=npfloat)
        return

    def step(self):
        """ Update the network """
        self.end_pairs()
        self.add_pairs()
        return

    def end_pairs(self):
        """ Remove expired edges """
        self.edge_dur -= 1
        active = self.edge_dur > 0
        self.p1 = self.p1[active]
        self.p2 = self.p2[active]
        self.edge_beta = self.edge_beta[active]
        self.edge_dur = self.edge_dur[active]
        return

    @staticmethod
    def get_source(inds, n_contacts):
        """ Get source array from contact counts """
        n_half_edges = n_contacts.sum()
        source = np.zeros(n_half_edges, dtype=int)
        count = 0
        for i, person_id in enumerate(inds):
            n = n_contacts[i]
            source[count:count + n] = person_id
            count += n
        return source

    def get_edges(self, inds, n_contacts):
        """ Create random edges by shuffling source into target """
        source = self.get_source(inds, n_contacts)
        target = np.random.permutation(source)
        return source, target

    def add_pairs(self):
        """ Generate new random edges """
        inds = np.arange(self.n_agents)
        n_conn = np.full(self.n_agents, self.n_contacts)

        # Calculate how many new edges are needed
        target_edges = n_conn.sum() / 2  # Divide by 2 since edges are bidirectional
        current_edges = len(self.p1)
        needed = target_edges - current_edges

        if needed > 0:
            # Scale down contacts proportionally to only create what's needed
            scale = needed / n_conn.sum()
            n_conn = np.round(n_conn * scale).astype(int)

            # Get the new edges
            p1, p2 = self.get_edges(inds, n_conn)
            beta = np.full(len(p1), self.beta, dtype=npfloat)
            if self.dur == 0:
                dur = np.zeros(len(p1), dtype=npfloat)
            else:
                dur = np.full(len(p1), self.dur, dtype=npfloat)

            # Append new edges
            self.p1 = np.concatenate([self.p1, p1])
            self.p2 = np.concatenate([self.p2, p2])
            self.edge_beta = np.concatenate([self.edge_beta, beta])
            self.edge_dur = np.concatenate([self.edge_dur, dur])
        return


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
        """ Progress the disease by one time step """
        self.step_state()
        self.update_immunity()
        self.infect()
        self.update_results()
        self.ti += 1
        return

    def step_state(self):
        """ Progress infectious -> recovered """
        recovered = (self.infected & (self.ti_recovered <= self.ti))
        self.infected[recovered] = False
        self.susceptible[recovered] = True    
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
    
    def infect(self):
        """ Calculate transmission """

        self.set_prognoses(uids)
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