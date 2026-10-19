"""
Imported from odin (seir_array_odin.R) by ss.load()

How each rate was read:
  S -> E           new_inf : S taken out of the flow; mass action with the number in I (not divided by the total: density-dependent, kept as counts) => beta * I.n / N
  R -> S           delta * R : R taken out of the flow; spontaneous => delta
  S ->             b * S : S taken out of the flow; an outflow => b
  -> S             Births : an inflow => Births
  E -> I           a dwell time of ss.constant(v=14.0)
  E ->             b * E : E taken out of the flow; an outflow => b
  I ->             b * I : I taken out of the flow; an outflow => b
  I -> R           sigma * I : I taken out of the flow; spontaneous => sigma
  R ->             b * R : R taken out of the flow; an outflow => b
  the flows of S from each level to the next at age_rate (1/365.0, 1460.0, 3650.0, 5475.0 days) read as aging: stratified by age, [0, 1, 5, 15, 30]
  the flows of E from each level to the next at age_rate (1/365.0, 1460.0, 3650.0, 5475.0 days) read as aging: stratified by age, [0, 1, 5, 15, 30]
  the flows of I from each level to the next at age_rate (1/365.0, 1460.0, 3650.0, 5475.0 days) read as aging: stratified by age, [0, 1, 5, 15, 30]
  the flows of R from each level to the next at age_rate (1/365.0, 1460.0, 3650.0, 5475.0 days) read as aging: stratified by age, [0, 1, 5, 15, 30]
  lag_inf = delay(new_inf * surv, 14) read as a fixed dwell time of 14 in E, whose outflows give the share who survive it (0.999233)
"""
import starsim as ss

# Parameters
Births = ss.par(ss.perday(547.945205479452))
N = ss.par(10000000.0)
b = ss.par(ss.perday(5.479452054794521e-05))
beta = ss.par(1.0)
delta = ss.par(ss.perday(0.016666666666666666))
dur = ss.par(ss.constant(v=ss.days(14.0)))
sigma = ss.par(ss.perday(0.03333333333333333))

# States
S, E, I, R = ss.states('S E I R')

# Transitions and outcomes
E_to_I   = ss.Transition(E, I, dur=dur, label='delay(new_inf * surv, 14)')
E_to_out = E >> ss.null @ b
I_to_R   = I >> R @ sigma
I_to_out = I >> ss.null @ b
R_to_S   = R >> S @ delta
R_to_out = R >> ss.null @ b
S_to_out = S >> ss.null @ b
in_to_S  = ss.null >> S @ Births
new_inf  = S >> E @ beta * I.n / N

seir_array_odin = ss.Disease(
    E_to_I, E_to_out, I_to_R, I_to_out, R_to_S, R_to_out,
    S_to_out, in_to_S, new_inf,
    init = dict(S=9999999, I=1),
    name = 'seir_array_odin',
)
seir_array_odin.stratify(age=[0.0, 1.0, 5.0, 15.0, 30.0], split=[0.0196078431372549, 0.07262164124909222, 0.15129508593560884, 0.17457125300262558, 0.5819041766754186], kind='age')

sim = ss.Sim(seir_array_odin, solver='ode', unit='day', stop=ss.days(365.0), total_pop=10000000)

# Not translated (W1601 ImportIncomplete):
#   lag_inf (history): for the first 14, odin's delay() gives the value at time 0 (as if the flow had been going on before the start), while the import starts with no one partway through E; a workaround is an inflow to I for the first 14 (ss.null >> I @ ss.timeseries(...))

if __name__ == '__main__':
    sim.run()
    sim.plot()
