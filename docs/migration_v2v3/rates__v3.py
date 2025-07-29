import starsim as ss

sir = ss.SIR(beta=ss.peryear(0.1))  # TODO: CHECK AUTOMATIC MIGRATION CHANGE FOR BETA
sis = ss.SIS(beta=ss.perday(0.001), waning=ss.peryear(0.05))  # TODO: CHECK AUTOMATIC MIGRATION CHANGE FOR BETA AND RATE
bir = ss.Births(birth_rate=ss.peryear(0.02))  # TODO: CHECK AUTOMATIC MIGRATION CHANGE FOR RATE
net = ss.MFNet(acts=ss.freqperyear(80))  # TODO: CHECK AUTOMATIC MIGRATION CHANGE FOR RATE
sim = ss.Sim(demographics=bir, diseases=[sir, sis], networks=net)