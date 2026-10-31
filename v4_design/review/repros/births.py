"""Newborns from ss.Births are in no state of a declared disease."""
import starsim as ss
sis = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> S', dur=ss.years(1)), init_prev=0.1, name='sis')
sim = ss.Sim(sis, n_agents=1000, demographics=[ss.Births(birth_rate=30), ss.Deaths(death_rate=10)], start=2000, stop=2020, verbose=0, rand_seed=1)
sim.run()
r = sim.results
print('n_alive', r.n_alive[-1], ' n_S+n_I', r.sis.n_S[-1] + r.sis.n_I[-1])
