import numpy as np
import starsim as ss

# Parameters
dur_critical = ss.par(ss.days(9.5))
dur_death = ss.par(ss.days(7.1))
dur_onset = ss.par(ss.days(3.2))
dur_recovery = ss.par(ss.days(7.1))
dur_severe = ss.par(ss.days(5.8))
dur_symptoms = ss.par(ss.days(2.0))
p_death = ss.par(0.3)
p_recovery = ss.par(0.7)

# States
S, E, INS, ISy, ISev, ICr, R = ss.states('S E INS ISy ISev ICr R')

# Transitions and outcomes
critical  = ss.Transition(ISev, [ICr, R], dur=dur_critical, p=[0.25, 0.75])
death     = ss.Transition(ICr, ss.null, dur=dur_death, p=[p_death])
infection = S >> E @ (seasonal(mean=0.05, amplitude=0.2, peak=-0.1643835616438356) * timeseries({0: 1.0, 29: 1.0, 30: 0.3}, mode='cosine') * (0.7 * INS + (0.35 - 0.1 * np.cos(3.14159 * np.minimum(np.maximum((0.455 * INS.n - 35) / 140, 0), 1))) * ISy))
onset     = E >> INS @ dur_onset
recovery  = ss.Transition(ICr, R, dur=dur_recovery, p=[p_recovery])
severe    = ss.Transition(ISy, [ISev, R], dur=dur_severe, p=[0.2, 0.8])
symptoms  = ss.Transition(INS, [ISy, R], dur=dur_symptoms, p=[0.91, 0.09])

secir = ss.Disease(
    critical, death, infection, onset, recovery, severe,
    symptoms,
    init = dict(E=100, INS=50, ISy=50, ISev=20, ICr=10, R=10),
    name = 'secir',
)
secir.stratify(age=['0', '1', '2'], split=[0.3333333333333333, 0.3333333333333333, 0.3333333333333333], kind='plain')

sim = ss.Sim(secir, unit='day', stop=ss.days(50.0), total_pop=10000)

# Not written as code (add by hand):
#   network "contacts" (MixingPools): add it to ss.Sim(networks=...)

if __name__ == '__main__':
    sim.run()
    sim.plot()
