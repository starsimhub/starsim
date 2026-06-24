"""
The dev/final toggle: same sim, two engines.

`run()`                -> fast Rust engine (statistical match, no CRN). Use for dev.
`run(engine='python')` -> the classic Python loop with common random numbers
                          (fully reproducible). Use for final/production runs.

This example times both and shows they agree on the epidemic size.
"""
import sciris as sc
import numpy as np
import starsim as ss
import starsim.rust as ssr


def make(n_agents=100_000, dur=100, seed=1, rust=True, label=None):
    if rust:
        SIS = ssr.SIS
        RNet = ssr.RandomNet
    else:
        SIS = ss.SIS
        RNet = ss.RandomNet
    return ss.Sim(diseases=SIS(beta=0.05, init_prev=0.01),
                  networks=RNet(n_contacts=10),
                  n_agents=n_agents, dur=dur, rand_seed=seed, verbose=0, label=label)

# Time the reproducible Python path (force it with engine='python')
with sc.timer('python') as t_py:
    py = make(rust=False, label='python').run(engine='python')

# Time the reproducible Python path (force it with engine='python')
with sc.timer('mixed') as t_mix:
    mix = make(rust=True, label='mixed').run(engine='python')

# Time the fast Rust engine
with sc.timer('rust  ') as t_rust:
    rust = make(rust=True, label='rust').run(engine='rust')


mix_speedup = t_py.elapsed / t_mix.elapsed
rust_speedup = t_py.elapsed / t_rust.elapsed
print(f'mixed speedup: {mix_speedup:.1f}x')
print(f'rust speedup: {rust_speedup:.1f}x')
for sim in [py, mix, rust]:
    res = sim.results.sis
    print(f'Sim {sim.label}: peak={res.n_infected.max():n}, final={res.n_infected[-1]:n}')
