"""Does a per-act beta on a sexual network depend on dt / unit?"""
import starsim as ss
def run(dt, unit='year', beta=0.05):
    si = ss.Disease(infection=ss.transmission('S -> I', beta=dict(mf=[beta, beta])), init_prev=0.05, name='si')
    sim = ss.Sim(si, networks=ss.MFNet(), n_agents=3000, start='2000-01-01', stop='2005-01-01', unit=unit, dt=dt, verbose=0, rand_seed=1)
    sim.run()
    return int(sim.results.si.infection.sum())
for dt in [ss.weeks(1), ss.months(1), ss.months(3)]:
    print(f'unit=year dt={dt}: infections {run(dt)}')
print('unit=month dt=1 month:', run(ss.months(1), unit='month'))
