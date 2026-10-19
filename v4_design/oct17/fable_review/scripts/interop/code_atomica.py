"""
Imported from Atomica (sir_framework.xlsx) by ss.load()

How each rate was read:
  sus -> inf       foi : kept as written, with counts => (1 - (1 - inf.n / (sus.n + inf.n + rec.n) * transpercontact) ** contacts * (1 - inf.n / (sus.n + inf.n + rec.n) * transpercontact * (contacts - contacts))) * (1 - susdeath)
  sus -> dead      susdeath : spontaneous => susdeath
  inf -> rec       1 / recrate : spontaneous => 1 / recrate
  inf -> dead      susdeath : spontaneous => susdeath
  inf -> dead      infdeath : spontaneous => infdeath
  rec -> dead      susdeath : spontaneous => susdeath
  ch_all           a result: ch_all
  ch_prev          a result: ch_prev
  ch_infrec        a result: ch_infrec
  ch_infsus        a result: ch_infsus
  ch_newinf        a result: ch_newinf
  ch_propinfrec    a result: ch_propinfrec
  ch_propinfsus    a result: ch_propinfsus
  ch_propnewinf    a result: ch_propnewinf
"""
import starsim as ss

# Parameters
contacts = ss.par(80.0, doc='Number of contacts annually')
infdeath = ss.par(ss.peryear(0.016), doc='Death rate for infected people')
recrate = ss.par(5.0, doc='Average duration of infections (years)')
susdeath = ss.par(ss.peryear(0.008), doc='Death rate for susceptible people')
transpercontact = ss.par(0.008, doc='Transmission probability per contact')

# States
sus = ss.State('sus', label='Susceptible')
inf = ss.State('inf', label='Infected')
rec = ss.State('rec', label='Recovered')
dead = ss.State('dead', label='Dead')

# Transitions and outcomes
inf_to_dead_infdeath = ss.Transition(inf, dead, rate=infdeath, label='Death rate for infected people')
inf_to_dead_susdeath = ss.Transition(inf, dead, rate=susdeath, label='Death rate for susceptible people')
inf_to_rec           = ss.Transition(inf, rec, rate=1 / recrate, label='Average duration of infections (years)')
rec_to_dead          = ss.Transition(rec, dead, rate=susdeath, label='Death rate for susceptible people')
sus_to_dead          = ss.Transition(sus, dead, rate=susdeath, label='Death rate for susceptible people')
sus_to_inf           = ss.Transition(sus, inf, rate=(1 - (1 - inf.n / (sus.n + inf.n + rec.n) * transpercontact) ** contacts * (1 - inf.n / (sus.n + inf.n + rec.n) * transpercontact * (contacts - contacts))) * (1 - susdeath), label='Force of infection')

SIR = ss.Disease(
    inf_to_dead_infdeath, inf_to_dead_susdeath, inf_to_rec, rec_to_dead, sus_to_dead, sus_to_inf,
    ch_all        = sus + inf + rec,
    ch_infrec     = inf + rec,
    ch_infsus     = inf + sus,
    ch_newinf     = sus + rec,
    ch_prev       = inf / (sus + inf + rec),
    ch_propinfrec = (inf + rec) / (sus + inf + rec),
    ch_propinfsus = (inf + sus) / (sus + inf + rec),
    ch_propnewinf = (sus + rec) / (sus + inf + rec),
    init          = dict(inf=285.71429, rec=14.285710000000714),
    name          = 'SIR',
)

sim = ss.Sim(SIR, solver='ode', integrator='euler', unit='year', start=2000.0, stop=2023.0, total_pop=1000.0000000000005, substeps=1)

if __name__ == '__main__':
    sim.run()
    sim.plot()
