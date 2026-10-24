import starsim as ss
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=10),
)
print(sir)
sim = ss.Sim(sir, stop=100).run()
print('--- repr', sim)
sim.plot()
import pylab as pl; pl.savefig('s2_plot.png')
print('--- describe after run')
sim.describe()
print('--- results keys', list(sim.results.keys()))
print(sim.results.disease.keys() if 'disease' in sim.results else '')
r = sim.results
print(type(r), r)
