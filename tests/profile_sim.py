import sciris as sc
import starsim as ss

kw = dict(n_agents=100e3, start=2000, dur=100, diseases='sis', networks='random')
sim = ss.Sim(**kw)

prof = ['profile', 'cprofile', 'time'][1]

if prof == 'profile':
    prf = sc.profile(run=sim.run, follow=[sim.run, ss.disease.Infection.infect, ss.network.RandomNet.step])

elif prof == 'cprofile':
    cprf = sc.cprofile(sort='selfpct', mintime=1e-3, maxitems=None, maxfunclen=None, maxpathlen=None)
    with cprf:
        sim.run()

else:
    with sc.timer() as T:
        sim.run()