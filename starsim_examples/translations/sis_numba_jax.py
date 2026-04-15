"""
Test simulation performance -- Numba version

Uses a single njit-compiled function with a for loop to avoid
per-step dispatch overhead. All state is carried as NumPy arrays (mutable).
"""
import numpy as np
from numba import njit
import sciris as sc


#%% Numba-compiled simulation loop

@njit
def numba_run(seed, infected, susceptible, ti_recovered, immunity, rel_sus,
              res_sus, res_inf, res_rel_sus,
              n_agents, n_edges, dur, source, edge_beta, disease_beta, dur_inf, waning, imm_boost):
    """ Run the full SIS simulation inside a single njit-compiled loop """
    np.random.seed(seed)

    # Pre-compute lognormal parameters (std=1)
    sigma = np.sqrt(np.log(1.0 + 1.0 / (dur_inf * dur_inf)))
    mu = np.log(dur_inf) - sigma * sigma / 2.0

    for ti in range(dur):
        ti_f = np.float32(ti)

        # Network: generate fresh random edges (uniform random targets)
        p1 = source
        p2 = np.random.randint(0, n_agents, size=n_edges).astype(np.int32)

        # Step state: infectious -> recovered
        for i in range(n_agents):
            if infected[i] and ti_recovered[i] <= ti_f:
                infected[i] = False
                susceptible[i] = True

        # Update immunity: wane and update relative susceptibility
        for i in range(n_agents):
            if immunity[i] > 0:
                immunity[i] *= (1.0 - waning)
                rel_sus[i] = max(0.0, 1.0 - immunity[i])

        # Infect: vectorized edge traversal and transmission
        is_target = np.zeros(n_agents, dtype=np.bool_)
        for e in range(n_edges):
            a = p1[e]
            b = p2[e]
            # p1 -> p2 transmission
            if infected[a] and susceptible[b]:
                beta_val = disease_beta * edge_beta[e] * rel_sus[b]
                if np.random.random() < beta_val:
                    is_target[b] = True
            # p2 -> p1 transmission
            if infected[b] and susceptible[a]:
                beta_val = disease_beta * edge_beta[e] * rel_sus[a]
                if np.random.random() < beta_val:
                    is_target[a] = True

        # Set prognoses for newly infected
        for i in range(n_agents):
            if is_target[i]:
                susceptible[i] = False
                infected[i] = True
                immunity[i] += imm_boost
                dur_sample = np.exp(np.random.normal() * sigma + mu)
                ti_recovered[i] = ti_f + dur_sample

        # Store results
        sus_count = np.float32(0)
        inf_count = np.float32(0)
        rel_sus_sum = np.float32(0)
        for i in range(n_agents):
            if susceptible[i]:
                sus_count += 1
            if infected[i]:
                inf_count += 1
            rel_sus_sum += rel_sus[i]
        res_sus[ti] = sus_count
        res_inf[ti] = inf_count
        res_rel_sus[ti] = rel_sus_sum / n_agents

    return infected, susceptible, ti_recovered, immunity, rel_sus, res_sus, res_inf, res_rel_sus


#%% Initialization and run

def make_initial_state(pars, seed=0, beta=0.05, init_prev=0.01, dur_inf=10, waning=0.05, imm_boost=1.0,
                       n_contacts=10, net_beta=1.0):
    """ Create initial simulation state (equivalent to class __init__ methods) """
    n_agents = pars.n_agents
    dur = pars.dur
    rng = np.random.default_rng(seed)

    # Disease state
    susceptible = np.ones(n_agents, dtype=np.bool_)
    infected = np.zeros(n_agents, dtype=np.bool_)
    ti_recovered = np.zeros(n_agents, dtype=np.float32)
    immunity = np.zeros(n_agents, dtype=np.float32)
    rel_sus = np.ones(n_agents, dtype=np.float32)

    # Seed initial infections
    n_inf = int(init_prev * n_agents)
    init_inds = rng.choice(n_agents, size=n_inf, replace=False)
    susceptible[init_inds] = False
    infected[init_inds] = True
    sigma = np.sqrt(np.log(1 + (1.0 / dur_inf) ** 2))
    mu = np.log(dur_inf) - sigma * sigma / 2.0
    dur_samples = np.exp(rng.normal(size=n_inf) * sigma + mu).astype(np.float32)
    ti_recovered[init_inds] = dur_samples

    # Result arrays
    res_sus = np.zeros(dur, dtype=np.float32)
    res_inf = np.zeros(dur, dtype=np.float32)
    res_rel_sus = np.zeros(dur, dtype=np.float32)

    # Network constants (edges regenerated each step; n_per_agent matches original scaling)
    n_per_agent = max(1, round(n_contacts / 2))
    source = np.repeat(np.arange(n_agents, dtype=np.int32), n_per_agent)
    edge_beta = np.full(n_agents * n_per_agent, net_beta, dtype=np.float32)

    n_edges = n_agents * n_per_agent
    state = (infected, susceptible, ti_recovered, immunity, rel_sus, res_sus, res_inf, res_rel_sus)
    return seed, state, n_edges, source, edge_beta, np.float32(beta), np.float32(dur_inf), np.float32(waning), np.float32(imm_boost)


def run_sim(pars, seed=0, **kwargs):
    """ Initialize and run a simulation """
    seed, state, n_edges, source, edge_beta, disease_beta, dur_inf, waning, imm_boost = make_initial_state(pars, seed=seed, **kwargs)
    infected, susceptible, ti_recovered, immunity, rel_sus, res_sus, res_inf, res_rel_sus = state
    result = numba_run(seed, infected, susceptible, ti_recovered, immunity, rel_sus,
                       res_sus, res_inf, res_rel_sus,
                       pars.n_agents, n_edges, pars.dur, source, edge_beta, disease_beta, dur_inf, waning, imm_boost)
    return result


# Define the parameters
pars = sc.dictobj(
    n_agents = 100_000,
    dur = 100,
)

# Warmup run to JIT compile
T = sc.timer()
result = run_sim(pars, seed=0)
T.toc('Warmup (includes JIT compilation)')

# Timed run
T = sc.timer()
result = run_sim(pars, seed=1)
T.toc(f'Time for SIS-Numba-Jax, n_agents={pars.n_agents}, dur={pars.dur}')


# Tests
def test_inf(result):
    n_inf = result[6]  # res_inf
    inf0 = n_inf[0]
    infm = n_inf.max()
    inf1 = n_inf[-1]
    tests = {
        f'Initial infections are nonzero: {inf0}' : inf0 > 0.005*pars.n_agents,
        f'Initial infections start low: {inf0}'   : inf0 < 0.05*pars.n_agents,
        f'Infections peak high: {infm}'           : infm > 0.5*pars.n_agents,
        f'Infections stabilize: {inf1}'           : inf1 < n_inf.max(),
    }
    for k,tf in tests.items():
        print(f'✓ {k}') if tf else print(f'× {k}')
    assert all(tests.values())
    return n_inf

n_inf = test_inf(result)
