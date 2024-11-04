import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss

do_plot = False

with sc.timer():
    ppl = ss.People(int(1e3))
    networks = ss.ndict(ss.MFNet(), ss.MaternalNet())

    hiv = ss.HIV()
    hiv.pars['beta'] = {'simple_sexual': [0.0008, 0.0004], 'maternal': [0.2, 0]}

    sim = ss.Sim(start=1950, stop=2050, people=ppl, networks=networks, demographics=ss.Pregnancy(), diseases=hiv)
    sim.init()
    sim.run()

if do_plot:
    plt.plot(sim.tivec, sim.results.hiv.n_infected)
    plt.show()
