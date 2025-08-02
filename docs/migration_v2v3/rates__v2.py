import starsim as ss

sir = ss.SIR(beta=ss.beta(0.1))
sis = ss.SIS(beta=ss.beta(0.001, 'days'), waning=ss.rate(0.05))
bir = ss.Births(birth_rate=ss.rate(0.02))
net = ss.MFNet(acts=ss.rate(80))
sim = ss.Sim(demographics=bir, diseases=[sir, sis], networks=net)