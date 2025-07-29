import starsim as ss

sir = ss.SIR(beta=ss.peryear(0.1))
sis = ss.SIS(beta=ss.perday(0.001))
sim = ss.Sim(diseases=[sir, sis], networks='random') 