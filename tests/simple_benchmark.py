import sciris as sc
import pylab as pl
import stisim as ss

do_plot = False

with sc.timer():
    ppl = ss.People(int(1e3))
    ppl.networks = ss.ndict(ss.mf(), ss.maternal())

    hiv = ss.HIV()
    hiv.pars['beta'] = {'simple_sexual': [0.0008, 0.0004], 'maternal': [0.2, 0]}
    
    sim = ss.Sim(start=1950, end=2050, people=ppl, demographics=ss.Pregnancy(), diseases=hiv)
    sim.initialize()
    sim.run()

if do_plot:
    pl.plot(sim.tivec, sim.results.hiv.n_infected)
    pl.show()
