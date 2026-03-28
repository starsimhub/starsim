"""
Test simulation performance -- JAX version

Uses a single JIT-compiled function with jax.lax.fori_loop to avoid
per-step dispatch overhead. All state is carried as JAX arrays (immutable).
"""
import os
os.environ["JAX_PLATFORM_NAME"] = ["cpu", "gpu"][1] # Toggle between CPU and GPU

import jax
import jax.numpy as jnp
from functools import partial
import sciris as sc

# Default dtype
jxfloat = jnp.float32

# Check where we're running
print(jax.devices())
print(jax.default_backend())


#%% JAX-compiled simulation loop

@partial(jax.jit, static_argnums=(2, 3, 4))
def jax_run(key, state, n_agents, n_edges, dur, source, edge_beta, disease_beta, dur_inf, waning, imm_boost):
    """ Run the full SIS simulation inside a single JIT-compiled loop """
    infected, susceptible, ti_recovered, immunity, rel_sus, res_sus, res_inf, res_rel_sus = state

    # Pre-compute lognormal parameters (std == mean simplification)
    sigma = jnp.sqrt(jnp.log(2.0))
    mu = jnp.log(dur_inf) - sigma * sigma / 2.0

    def step_fn(ti, carry):
        key, infected, susceptible, ti_recovered, immunity, rel_sus, res_sus, res_inf, res_rel_sus = carry
        ti_f = jnp.float32(ti)
        key, key_net, key_inf, key_prog = jax.random.split(key, 4)

        # Network: generate fresh random edges (uniform random targets)
        p1 = source
        p2 = jax.random.randint(key_net, shape=(n_edges,), minval=0, maxval=n_agents, dtype=jnp.int32)

        # Step state: infectious -> recovered
        recovered = infected & (ti_recovered <= ti_f)
        infected = jnp.where(recovered, False, infected)
        susceptible = jnp.where(recovered, True, susceptible)

        # Update immunity: wane and update relative susceptibility
        has_imm = immunity > 0
        immunity = jnp.where(has_imm, immunity * (1.0 - waning), immunity)
        rel_sus = jnp.where(has_imm, jnp.maximum(0.0, 1.0 - immunity), rel_sus)

        # Infect: vectorized edge traversal and transmission
        p1_can = infected[p1] & susceptible[p2]
        beta_p1 = disease_beta * edge_beta * rel_sus[p2] * p1_can
        p2_can = infected[p2] & susceptible[p1]
        beta_p2 = disease_beta * edge_beta * rel_sus[p1] * p2_can

        key1, key2 = jax.random.split(key_inf)
        transmit_p1 = jax.random.uniform(key1, shape=p1.shape) < beta_p1
        transmit_p2 = jax.random.uniform(key2, shape=p2.shape) < beta_p2

        is_target = jnp.zeros(n_agents, dtype=bool)
        is_target = is_target.at[p2].max(transmit_p1)  # max = OR for booleans
        is_target = is_target.at[p1].max(transmit_p2)

        # Set prognoses for newly infected
        susceptible = jnp.where(is_target, False, susceptible)
        infected = jnp.where(is_target, True, infected)
        immunity = jnp.where(is_target, immunity + imm_boost, immunity)
        normal_samples = jax.random.normal(key_prog, shape=(n_agents,)) * sigma + mu
        dur_samples = jnp.exp(normal_samples)
        ti_recovered = jnp.where(is_target, ti_f + dur_samples, ti_recovered)

        # Store results
        res_sus = res_sus.at[ti].set(jnp.sum(susceptible))
        res_inf = res_inf.at[ti].set(jnp.sum(infected))
        res_rel_sus = res_rel_sus.at[ti].set(jnp.mean(rel_sus))

        return (key, infected, susceptible, ti_recovered, immunity, rel_sus, res_sus, res_inf, res_rel_sus)

    init = (key, infected, susceptible, ti_recovered, immunity, rel_sus, res_sus, res_inf, res_rel_sus)
    final = jax.lax.fori_loop(0, dur, step_fn, init)
    return final[1:]  # Drop key, return state + results


#%% Initialization and run

def make_initial_state(pars, seed=0, beta=0.05, init_prev=0.01, dur_inf=10, waning=0.05, imm_boost=1.0,
                       n_contacts=10, net_beta=1.0):
    """ Create initial simulation state (equivalent to class __init__ methods) """
    n_agents = pars.n_agents
    dur = pars.dur
    key = jax.random.PRNGKey(seed)

    # Disease state
    susceptible = jnp.ones(n_agents, dtype=bool)
    infected = jnp.zeros(n_agents, dtype=bool)
    ti_recovered = jnp.zeros(n_agents, dtype=jxfloat)
    immunity = jnp.zeros(n_agents, dtype=jxfloat)
    rel_sus = jnp.ones(n_agents, dtype=jxfloat)

    # Seed initial infections
    n_inf = int(init_prev * n_agents)
    key, subkey = jax.random.split(key)
    init_inds = jax.random.choice(subkey, n_agents, shape=(n_inf,), replace=False)
    susceptible = susceptible.at[init_inds].set(False)
    infected = infected.at[init_inds].set(True)

    # Result arrays
    res_sus = jnp.zeros(dur, dtype=jxfloat)
    res_inf = jnp.zeros(dur, dtype=jxfloat)
    res_rel_sus = jnp.zeros(dur, dtype=jxfloat)

    # Network constants (edges regenerated each step; n_per_agent matches original scaling)
    n_per_agent = max(1, round(n_contacts / 2))
    source = jnp.repeat(jnp.arange(n_agents, dtype=jnp.int32), n_per_agent)
    edge_beta = jnp.full(n_agents * n_per_agent, net_beta, dtype=jxfloat)

    n_edges = n_agents * n_per_agent
    state = (infected, susceptible, ti_recovered, immunity, rel_sus, res_sus, res_inf, res_rel_sus)
    return key, state, n_edges, source, edge_beta, jnp.float32(beta), jnp.float32(dur_inf), jnp.float32(waning), jnp.float32(imm_boost)


def run_sim(pars, seed=0, **kwargs):
    """ Initialize and run a simulation """
    key, state, n_edges, source, edge_beta, disease_beta, dur_inf, waning, imm_boost = make_initial_state(pars, seed=seed, **kwargs)
    result = jax_run(key, state, pars.n_agents, n_edges, pars.dur, source, edge_beta, disease_beta, dur_inf, waning, imm_boost)
    return result


# Define the parameters
pars = sc.dictobj(
    n_agents = 100_000,
    dur = 100,
)

# Warmup run to JIT compile
T = sc.timer()
result = run_sim(pars, seed=0)
jax.block_until_ready(result)
T.toc('Warmup (includes JIT compilation)')

# Timed run
T = sc.timer()
result = run_sim(pars, seed=1)
jax.block_until_ready(result)
T.toc(f'Time for SIS-JAX, n_agents={pars.n_agents}, dur={pars.dur}')
