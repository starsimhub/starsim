"""
Imported from EMULSION (compart_SIR.yaml) by ss.load()

How each rate was read:
  S -> I           force_of_infection : exposure to I (divided by the total: frequency-dependent) => transmission_I * I
  I -> R           recovery : spontaneous => recovery
  percentage_prevalence a result: percentage_prevalence
  total_population a result: total_population
"""
import starsim as ss

# Parameters
recovery = ss.par(ss.perday(0.1), doc='recovery rate (/day)')
transmission_I = ss.par(ss.perday(0.5), doc='transmission rate from infectious individuals (/day)')

# States
S = ss.State('S', label='Susceptible')
I = ss.State('I', label='Infectious')
R = ss.State('R', label='Resistant')

# Transitions and outcomes
I_to_R             = I >> R @ recovery
force_of_infection = S >> I @ transmission_I * I

compart_SIR = ss.Disease(
    I_to_R, force_of_infection,
    percentage_prevalence = 100 * I / (S + I + R),
    total_population      = S + I + R,
    init                  = dict(S=99, I=1),
    name                  = 'compart_SIR',
)

sim = ss.Sim(compart_SIR, solver='binomial', unit='day', stop=ss.days(100.0), total_pop=100)

if __name__ == '__main__':
    sim.run()
    sim.plot()
