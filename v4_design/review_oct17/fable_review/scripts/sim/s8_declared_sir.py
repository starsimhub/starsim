import starsim as ss, sciris as sc, numpy as np
sc.heading('G: ss.SIR as a declared class')
class SIRd(ss.Disease):
    beta = ss.par(ss.peryear(0.1))
    S = ss.State(); I = ss.State(); R = ss.State(); D = ss.State()
    infection = S >> I @ beta * I
    recovery = ss.progression('I -> R | D', dur=ss.lognorm_ex(mean=ss.years(6), std=ss.years(1)), p=[0.99, 0.01])
s = ss.Sim(SIRd(init_prev=0.01), networks=ss.RandomNet(), n_agents=1000, start=2000, stop=2020, verbose=0)
s.init(); print(s.solver, s.solver_reason); s.run(); print(list(s.results.sird.keys()))
s2 = ss.Sim(SIRd(init_prev=0.01), start=2000, stop=2020, verbose=0).run(); print(s2.solver, s2.notes)
print('declared n_I peak (abm, random net)', s.results.sird.n_I.max(), '| ode homogeneous', s2.results.sird.n_I.max())
s3 = ss.Sim(ss.SIR(), networks=ss.RandomNet(), n_agents=1000, start=2000, stop=2020, verbose=0).run()
print('coded ss.SIR n_infected peak', s3.results.sir.n_infected.max())
s2.describe()
sc.heading('H: People repr in ODE sim')
try: print(repr(s2.people))
except Exception as E: print('ERROR', type(E).__name__, str(E)[:120])
sc.heading('I: n_agents repr for ODE')
print(repr(s2), '| pars.n_agents', s2.pars.n_agents, '| len(people)', len(s2.people))
