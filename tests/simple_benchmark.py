import sciris as sc
import pylab as pl
import stisim as sti

do_plot = False

with sc.timer():
    ppl = sti.People(int(1e3))
    ppl.networks = sti.ndict(sti.simple_sexual(), sti.maternal())
    
    hiv = sti.HIV()
    hiv.pars['beta'] = {'simple_sexual': [0.0008, 0.0004], 'maternal': [0.2, 0]}
    
    sim = sti.Sim(start=1950, end=2050, people=ppl, modules=[hiv, sti.Pregnancy()])
    sim.initialize()
    sim.run()

if do_plot:
    pl.plot(sim.tivec, sim.results.hiv.n_infected)
    pl.show()