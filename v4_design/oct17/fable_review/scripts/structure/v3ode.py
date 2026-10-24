import starsim as ss
sim = ss.Sim(diseases=ss.SIR(), networks='random', n_agents=500, dur=2, solver='ode', verbose=1).run()
print('solver attr:', getattr(sim, 'solver', None), '| pars.solver:', sim.pars.get('solver'), '| engine:', type(getattr(sim, 'engine', None)).__name__, '| reason:', sim.solver_reason)
