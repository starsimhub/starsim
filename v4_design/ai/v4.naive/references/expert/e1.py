import starsim as ss
# T1 expert
S, I, R = ss.states(3)
beta = 2.5/7
infection = S >> I @ beta * I
recovery = I >> R @ ss.days(7)
sir = ss.Disease(infection, recovery, init_prev=10/1e5)
sim = ss.Sim(sir, total_pop=1e5, stop=ss.days(365), verbose=1).run()
r = sim.results.sir
print('T1', 1 - r.n_S[-1]/1e5)
