import sciris as sc
import matplotlib.pyplot as pl
import starsim as ss

do_plot = False

with sc.timer():
    ppl = ss.People(int(1e3))
    networks = ss.ndict(ss.MFNet(), ss.MaternalNet())

    hiv = ss.HIV()
    hiv.pars['beta'] = {'simple_sexual': [0.0008, 0.0004], 'maternal': [0.2, 0]}
    
    sim = ss.Sim(start=1950, end=2050, people=ppl, networks=networks, demographics=ss.Pregnancy(), diseases=hiv)
    sim.init()
    sim.run()

if do_plot:
    pl.plot(sim.tivec, sim.results.hiv.n_infected)
    pl.show()
