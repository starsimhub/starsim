import starsim as ss

sir = ss.SIR(beta=ss.peryear(0.1))  # TODO: CHECK AUTOMATIC MIGRATION CHANGE
sis = ss.SIS(beta=ss.perday(0.001))  # TODO: CHECK AUTOMATIC MIGRATION CHANGE
sim = ss.Sim(diseases=[sir, sis], networks='random')