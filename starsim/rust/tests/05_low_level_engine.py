"""
Calling the engine directly (no ss.Sim).

The bridge (ss.Sim + ssr modules) is the convenient front end, but the engine
itself takes a plain spec: lists of (module_name, params) for networks, diseases,
and demographics. This is the level the bridge dispatches to, and it's handy for
sweeps or when you want full control over the parameters passed to Rust.

Note: these are the engine's *effective* per-step parameters (e.g. beta is a
per-contact probability, dur_inf is in timesteps) -- the bridge computes these
for you from the Starsim modules.
"""
import numpy as np
import ssr_engine

res = ssr_engine.run(
    n_agents = 100_000,
    n_steps  = 100,
    seed     = 1,
    networks = [('randomnet', {'n_contacts': 10.0, 'dur': 0.0, 'beta': 1.0})],
    diseases = [('sis', {'beta': 0.049, 'init_prev': 0.01,
                         'dur_inf': 10.0, 'dur_inf_std': 1.0,
                         'waning': 0.0, 'imm_boost': 0.0})],
    demographics = [('births', {'birth_prob': 0.01}),   # balanced with deaths -> stable population
                    ('deaths', {'death_prob': 0.01})],
)

# Results come back as a dict of numpy arrays keyed '<module>_<result>'
print('result keys:', sorted(res.keys()))
ni = np.asarray(res['sis_n_infected'])
print(f'n_infected: peak={ni.max():.0f}, end={ni[-1]:.0f}')
