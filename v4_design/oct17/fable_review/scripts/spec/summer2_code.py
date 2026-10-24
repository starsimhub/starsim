import starsim as ss

# Parameters
beta = ss.par(2.0)
rate_incidence = ss.par(0.5)
rate_infection_death = ss.par(0.05)
rate_recovery = ss.par(0.3)

# States
S, E, I, R = ss.states('S E I R')

# Transitions and outcomes
incidence       = E >> I @ rate_incidence
infection       = S >> E @ beta * I
infection_death = I >> ss.null @ rate_infection_death
recovery        = I >> R @ rate_recovery

seir = ss.Disease(
    incidence, infection, infection_death, recovery,
    init = dict(I=ss.by([[5.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]], dims=['age', 'clinical'])),
    name = 'seir',
)
seir.stratify(age=['young', 'old'], split=[0.5, 0.5], adjust={'incidence': [1.0, 1.1]}, kind='plain')
seir.stratify(clinical=['asymptomatic', 'symptomatic', 'isolated', 'hospital'], split=ss.by([[0.4, 0.3, 0.2, 0.1], [0.3, 0.3, 0.2, 0.3]], dims=['age', 'clinical']), adjust={'infection_death': [0.0, 1.0, 1.0, 2.0], 'rel_trans': [0.5, 1.0, 0.2, 0.1]}, kind='plain', states=['I'])

sim = ss.Sim(seir, start=0.0, stop=20.0, dt=0.1, total_pop=1000)

if __name__ == '__main__':
    sim.run()
    sim.plot()
