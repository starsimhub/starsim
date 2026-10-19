"""
Imported from flepiMoP (config_sample_2pop.yml) by ss.load()

How each rate was read:
  S -> E           (Ro * gamma) * S * I / __N__ : S taken out of the flow; exposure to I (divided by the total: frequency-dependent) => Ro * gamma * I
  E -> I           (sigma) * E : E taken out of the flow; spontaneous => sigma
  I -> R           (gamma) * I : I taken out of the flow; spontaneous => gamma
  subpop: commuting as visits, each traveler away for half the day (flepiMoP's percent_day_away)
"""
import starsim as ss

# Parameters
Ro = ss.par(ss.perday(2.5))
gamma = ss.par(ss.perday(0.2))
sigma = ss.par(ss.perday(0.25))

# States
S, E, I, R = ss.states('S E I R')

# Transitions and outcomes
E_to_I = E >> I @ sigma
I_to_R = I >> R @ gamma
S_to_E = S >> E @ Ro * gamma * I

sample_2pop = ss.Disease(
    E_to_I, I_to_R, S_to_E,
    init = dict(E=ss.by(subpop=[5.0, 0.0])),
    name = 'sample_2pop',
)
sample_2pop.stratify(subpop=['large_province', 'small_province'], split=[0.9, 0.1], kind='space', mobility=ss.Mobility(kind='visit', matrix=[[0.0, 0.005555555555555556], [0.01, 0.0]]))

sim = ss.Sim(sample_2pop, solver='ode', unit='day', start='2020.02.01', stop='2020.08.31', total_pop=10000)

if __name__ == '__main__':
    sim.run()
    sim.plot()
