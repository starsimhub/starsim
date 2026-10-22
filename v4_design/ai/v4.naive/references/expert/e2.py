import numpy as np, starsim as ss
ss.options(verbose=0) if hasattr(ss.options,'verbose') else None
out = {}
# T2 SEIR
S, E, I, R = ss.states(4)
infection = S >> E @ (3/7) * I
latent = E >> I @ ss.days(5)
recovery = I >> R @ ss.days(7)
seir = ss.Disease(infection, latent, recovery, init_prev=100/1e6)
sim = ss.Sim(seir, total_pop=1e6, stop=ss.days(365)).run()
n_I = sim.results.seir.n_I
out['T2'] = (n_I.argmax(), n_I.max())
# T3 SIRS
S, I, R = ss.states(3)
infection = S >> I @ (3/5) * I
recovery = I >> R @ ss.days(5)
waning = R >> S @ ss.days(365)
sirs = ss.Disease(infection, recovery, waning, init_prev=0.01)
sim = ss.Sim(sirs, total_pop=1e5, stop=ss.years(20)).run()
r = sim.results.sirs
out['T3'] = r.n_I[-1]/(r.n_S[-1]+r.n_I[-1]+r.n_R[-1])
print(out)
