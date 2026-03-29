"""
Test simulation performance -- Numba-optimized version
"""
import numpy as np
import numba as nb
import sciris as sc

# Default dtype
npfloat = np.float32

def floatzeros(*args, **kwargs):
    """ Create an array with the default dtype """
    return np.zeros(*args, dtype=npfloat, **kwargs)

def boolzeros(*args, **kwargs):
    """ Create a boolean array """
    return np.zeros(*args, dtype=bool, **kwargs)


#%% Numba-optimized kernels

# NB: fastmath and parallel make little difference
jit_kw = dict(cache=True, fastmath=False, parallel=False)

@nb.njit(**jit_kw)
def nb_get_edges(inds, n_contacts):
    """ Build source array from contact counts and Fisher-Yates shuffle into target """
    n_half_edges = 0
    for i in range(len(n_contacts)):
        n_half_edges += n_contacts[i]

    source = np.empty(n_half_edges, dtype=nb.int64)
    count = 0
    for i in range(len(inds)):
        n = n_contacts[i]
        for j in range(n):
            source[count + j] = inds[i]
        count += n

    # Fisher-Yates shuffle for target
    target = source.copy()
    n = len(target)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        target[i], target[j] = target[j], target[i]

    return source, target


@nb.njit(**jit_kw)
def nb_end_pairs(p1, p2, edge_beta, edge_dur):
    """ Decrement durations and compact active edges in fused passes """
    n = len(p1)

    # First pass: decrement and count active
    n_active = 0
    for i in range(n):
        edge_dur[i] -= 1
        if edge_dur[i] > 0:
            n_active += 1

    # Second pass: compact into new arrays
    new_p1 = np.empty(n_active, dtype=p1.dtype)
    new_p2 = np.empty(n_active, dtype=p2.dtype)
    new_beta = np.empty(n_active, dtype=edge_beta.dtype)
    new_dur = np.empty(n_active, dtype=edge_dur.dtype)
    j = 0
    for i in range(n):
        if edge_dur[i] > 0:
            new_p1[j] = p1[i]
            new_p2[j] = p2[i]
            new_beta[j] = edge_beta[i]
            new_dur[j] = edge_dur[i]
            j += 1

    return new_p1, new_p2, new_beta, new_dur


@nb.njit(**jit_kw)
def nb_step_state(infected, susceptible, ti_recovered, ti):
    """ Progress infectious -> recovered in one fused pass """
    for i in range(len(infected)):
        if infected[i] and ti_recovered[i] <= ti:
            infected[i] = False
            susceptible[i] = True


@nb.njit(**jit_kw)
def nb_update_immunity(immunity, rel_sus, waning):
    """ Wane immunity and update relative susceptibility in one pass; return mean rel_sus """
    total = 0.0
    n = len(immunity)
    wane_factor = 1.0 - waning
    for i in range(n):
        if immunity[i] > 0:
            immunity[i] *= wane_factor
            rs = 1.0 - immunity[i]
            if rs < 0:
                rs = 0.0
            rel_sus[i] = rs
        total += rel_sus[i]
    return total / n


@nb.njit(**jit_kw)
def nb_infect(p1, p2, edge_beta, infected, susceptible, rel_sus, disease_beta, n_agents):
    """ Fused edge traversal, transmission sampling, and deduplication """
    n_edges = len(p1)

    # Boolean flag array for O(1) deduplication
    is_target = np.zeros(n_agents, dtype=np.bool_)

    for i in range(n_edges):
        a = p1[i]
        b = p2[i]
        eb = edge_beta[i]

        # p1 infected -> p2 susceptible
        if infected[a] and susceptible[b]:
            beta = disease_beta * eb * rel_sus[b]
            if np.random.random() < beta:
                is_target[b] = True

        # p2 infected -> p1 susceptible
        if infected[b] and susceptible[a]:
            beta = disease_beta * eb * rel_sus[a]
            if np.random.random() < beta:
                is_target[a] = True

    # Collect unique targets
    n_targets = 0
    for i in range(n_agents):
        if is_target[i]:
            n_targets += 1

    uids = np.empty(n_targets, dtype=nb.int64)
    j = 0
    for i in range(n_agents):
        if is_target[i]:
            uids[j] = i
            j += 1

    return uids


@nb.njit(**jit_kw)
def nb_set_prognoses(uids, susceptible, infected, immunity, ti_recovered, imm_boost, dur_inf_mean, ti):
    """ Set prognoses with inline lognormal sampling """
    sigma = np.sqrt(np.log(2.0))  # Simplifies when std == mean
    mu = np.log(dur_inf_mean) - sigma * sigma / 2.0

    for i in range(len(uids)):
        uid = uids[i]
        susceptible[uid] = False
        infected[uid] = True
        immunity[uid] += imm_boost
        dur = np.random.lognormal(mu, sigma)
        ti_recovered[uid] = ti + dur


@nb.njit(**jit_kw)
def nb_count_states(susceptible, infected):
    """ Count susceptible and infected in a single pass """
    n_sus = 0
    n_inf = 0
    for i in range(len(susceptible)):
        n_sus += susceptible[i]
        n_inf += infected[i]
    return n_sus, n_inf


#%% Simulation classes

# Define the parameters
pars = sc.dictobj(
    n_agents = 100_000,
    dur = 100,
)

class Sim(sc.prettyobj):
    """ Minimal translation of Starsim's Sim class """
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
        dis.step(net)
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

        # Initialize edge arrays (use int64 for Numba compatibility)
        self.p1 = np.empty(0, dtype=np.int64)
        self.p2 = np.empty(0, dtype=np.int64)
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
        if len(self.p1) > 0:
            self.p1, self.p2, self.edge_beta, self.edge_dur = \
                nb_end_pairs(self.p1, self.p2, self.edge_beta, self.edge_dur)
        return

    def add_pairs(self):
        """ Generate new random edges """
        inds = np.arange(self.n_agents, dtype=np.int64)
        n_conn = np.full(self.n_agents, self.n_contacts, dtype=np.int64)

        # Calculate how many new edges are needed
        target_edges = n_conn.sum() / 2  # Divide by 2 since edges are bidirectional
        current_edges = len(self.p1)
        needed = target_edges - current_edges

        if needed > 0:
            # Scale down contacts proportionally to only create what's needed
            scale = needed / n_conn.sum()
            n_conn = np.round(n_conn * scale).astype(np.int64)

            # Get the new edges via Numba
            p1, p2 = nb_get_edges(inds, n_conn)
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
        self.rel_sus = np.ones(self.n_agents, dtype=npfloat)

        # Seed initial infections
        self.susceptible[:] = True
        n_inf = int(self.init_prev * self.n_agents)
        init_inds = np.random.choice(self.n_agents, n_inf, replace=False)
        self.susceptible[init_inds] = False
        self.infected[init_inds] = True
        self.ti = 0

        # Create results
        self.n_susceptible = floatzeros(self.dur)
        self.n_infected = floatzeros(self.dur)
        self.mean_rel_sus = floatzeros(self.dur)
        return

    def step(self, net):
        """ Progress the disease by one time step """
        self.step_state()
        self.update_immunity()
        self.infect(net)
        self.update_results()
        self.ti += 1
        return

    def step_state(self):
        """ Progress infectious -> recovered """
        nb_step_state(self.infected, self.susceptible, self.ti_recovered, self.ti)
        return

    def update_immunity(self):
        self.mean_rel_sus[self.ti] = nb_update_immunity(self.immunity, self.rel_sus, self.waning)
        return

    def update_results(self):
        n_sus, n_inf = nb_count_states(self.susceptible, self.infected)
        self.n_susceptible[self.ti] = n_sus
        self.n_infected[self.ti] = n_inf
        return

    def infect(self, net):
        """ Calculate transmission across network edges """
        uids = nb_infect(net.p1, net.p2, net.edge_beta,
                         self.infected, self.susceptible, self.rel_sus,
                         self.beta, self.n_agents)
        if len(uids):
            nb_set_prognoses(uids, self.susceptible, self.infected, self.immunity,
                             self.ti_recovered, self.imm_boost,
                             float(self.dur_inf), float(self.ti))
        return


# Create the sim
sim = Sim(
    pars = pars,
    disease = SIS(pars),
    network = RandomNet(pars),
    verbose = 0,
)

# Warmup run to JIT compile all Numba functions
T = sc.timer()
sim.run()
T.toc('Warmup (includes JIT compilation)')

# Create fresh sim for timed run
sim = Sim(
    pars = pars,
    disease = SIS(pars),
    network = RandomNet(pars),
    verbose = 0,
)

# Timed run
T = sc.timer()
sim.run()
T.toc(f'Time for SIS-Numba, n_agents={pars.n_agents}, dur={pars.dur}')

# Tests
def test_inf(sim):
    n_inf = sim.disease.n_infected
    inf0 = n_inf[0]
    infm = n_inf.max()
    inf1 = n_inf[-1]
    tests = {
        f'Initial infections are nonzero: {inf0}' : inf0 > 0.01*pars.n_agents,
        f'Initial infections start low: {inf0}'   : inf0 < 0.05*pars.n_agents,
        f'Infections peak high: {infm}'           : infm > 0.5*pars.n_agents,
        f'Infections stabilize: {inf1}'           : inf1 < n_inf.max(),
    }
    for k,tf in tests.items():
        print(f'✓ {k}') if tf else print(f'× {k}')
    assert all(tests.values())
    return tests

test_inf(sim)
