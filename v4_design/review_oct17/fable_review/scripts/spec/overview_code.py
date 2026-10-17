import starsim as ss

# Parameters
beta = ss.par(ss.perday(0.05), bounds=[0.01, 0.5])
dur = ss.par(ss.days(10.0))

# States
S, I, R = ss.states('S I R')

# Transitions and outcomes
infection = S >> I @ beta * I
recovery  = I >> R @ dur
cases     = ss.outcome(0.5 * infection, delay=ss.days(5.0), every=ss.weeks(1.0))

sir = ss.Disease(
    infection, recovery, cases,
    init_prev = 0.01,
    name      = 'sir',
)

sim = ss.Sim(sir, unit='day', start='2020.01.01', stop='2021.01.01', total_pop=50000000, networks=[ss.contacts('Kenya')], interventions=[ss.modify('sir.pars.beta', multiply=0.6, start='2020-03-15', name='masks')])
sim.stratify(age=[0.0, 15.0, 65.0], split=[0.15, 0.5, 0.35], kind='age')

if __name__ == '__main__':
    sim.run()
    sim.plot()
