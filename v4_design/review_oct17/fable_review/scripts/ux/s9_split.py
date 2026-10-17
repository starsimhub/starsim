import starsim as ss
S, E, I, R, D = ss.states(5)
seir = ss.Disease(S >> E @ 0.3*I, E >> I @ ss.days(5), I >> (0.9*R + 0.1*D) @ ss.days(10), init_prev=0.001)
seir.stratify(vaccinated=[False, True], adjust={'infection': [1.0, 0.2]})
sim = ss.Sim(seir, start='2020-01-01', stop='2020-02-01', total_pop=1e6, verbose=0).run()
r = sim.results.disease.n_S
print('n_S at t=0: unvaccinated', r.select(vaccinated=False).values[0], 'vaccinated', r.select(vaccinated=True).values[0])
print('name of the disease:', seir.name, '| transitions:', [t.name for t in seir.transitions] if hasattr(seir,'transitions') else None)
