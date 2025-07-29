import starsim as ss

sir = ss.SIR(beta=ss.peryear(0.1))  # TODO: Check automatic migration change for ss.beta
sis = ss.SIS(beta=ss.perday(0.001), waning=ss.rate(0.05))  # TODO: Check automatic migration change for ss.beta
bir = ss.Births(birth_rate=ss.peryear(0.02))  # TODO: Check automatic migration change for ss.rate
net = ss.MFNet(acts=ss.freqperyear(80))  # TODO: Check automatic migration change for ss.rate
sim = ss.Sim(demographics=bir, diseases=[sir, sis], networks=net)