import starsim as ss
import starsim_examples as sse

hiv = sse.HIV()
net = sse.DiskNet()
sim = ss.Sim(diseases=hiv, networks=net).run()