import sciris as sc
import starsim as ss

def make_sim():
    kw = dict(n_agents=100e3, start=2000, dur=100, diseases='sis', networks='random', demographics=True)
    sim = ss.Sim(**kw)
    return sim

to_run = [
    'profile',
    'cprofile',
    'time'
    'plot_cpu',
]

if 'profile' in to_run:
    sim = make_sim()
    prf = sc.profile(run=sim.run, follow=[sim.run])

elif 'cprofile' in to_run:
    sim = make_sim()
    cprf = sc.cprofile(sort='selfpct', mintime=1e-3, maxitems=None, maxfunclen=None, maxpathlen=None)
    with cprf:
        sim.run()

elif 'time' in to_run:
    sim = make_sim()
    with sc.timer() as T:
        sim.run()

if 'plot_cpu' in to_run:
    sim.loop.plot_cpu()