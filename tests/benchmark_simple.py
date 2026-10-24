import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss
import starsim.library as ssl

do_plot = False

with sc.timer():
    ppl = ss.People(int(1e3))
    networks = ss.ndict(ss.SexualNet(), ss.PrenatalNet())

    hiv = ssl.diseases.HIV()
    hiv.pars['beta'] = {'sexual': [0.0008, 0.0004], 'maternal': [0.2, 0]}

    sim = ss.Sim(start=1950, stop=2050, people=ppl, networks=networks, demographics=ss.Pregnancy(), diseases=hiv)
    sim.init()
    sim.run()

if do_plot:
    plt.plot(sim.tivec, sim.results.hiv.infected)
    plt.show()
