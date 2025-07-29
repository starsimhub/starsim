import starsim as ss

sir = ss.SIR(beta=ss.beta(0.1))
sis = ss.SIS(beta=ss.beta(0.001, 'days'))
sim = ss.Sim(diseases=[sir, sis], networks='random')