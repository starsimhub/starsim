import sciris as sc
import starsim as ss

kw = dict(n_agents=10e3, start=2000, dur=100, diseases='sis', networks='random', demographics=True)
sim = ss.Sim(**kw)

prof = ['profile', 'cprofile', 'time'][1]

if prof == 'profile':
    prf = sc.profile(run=sim.run, follow=[sim.run])

elif prof == 'cprofile':
    cprf = sc.cprofile(sort='selfpct', mintime=1e-3, maxitems=None, maxfunclen=None, maxpathlen=None)
    with cprf:
        sim.run()

else:
    with sc.timer() as T:
        sim.run()

sim.loop.plot_cpu()