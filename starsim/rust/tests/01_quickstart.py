"""
Quickstart: build a sim from Rust-backed modules and run it.

This is the headline API. `ssr.SIS()` and `ssr.RandomNet()` look and behave like
normal Starsim modules, but because the whole sim is built from them, `sim.run()`
automatically runs the entire loop in Rust -- typically ~3x faster at 100k agents,
with results matching the pure-Python sim statistically.
"""
import starsim as ss
import starsim.rust as ssr

# Build exactly as you would a normal Starsim sim -- just with ssr modules
sim = ss.Sim(
    diseases = ssr.SIS(beta=0.05, init_prev=0.01),
    networks = ssr.RandomNet(n_contacts=10),
    n_agents = 100_000,
    dur      = 100,
    verbose  = 0,
)

# Runs on the native Rust engine (auto-detected because every module is ssr-native).
# sim.results is populated just like a normal run, so plotting/analysis work.
sim.run()

res = sim.results.sis
print(f'n_infected: start={res.n_infected[0]:.0f}, '
      f'peak={res.n_infected.max():.0f}, end={res.n_infected[-1]:.0f}')

# Optional: plot (off by default so the example runs headless)
do_plot = False
if do_plot:
    import matplotlib.pyplot as plt
    plt.plot(res.n_susceptible, label='susceptible')
    plt.plot(res.n_infected, label='infected')
    plt.legend(); plt.xlabel('Timestep'); plt.ylabel('People'); plt.show()
