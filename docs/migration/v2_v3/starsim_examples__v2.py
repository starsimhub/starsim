import starsim as ss

hiv = ss.HIV()
net = ss.DiskNet()
sim = ss.Sim(diseases=hiv, networks=net).run()