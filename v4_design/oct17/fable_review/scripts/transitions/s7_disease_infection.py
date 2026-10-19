"""ss.Disease: the declared model next to Starsim 3's Infection."""
import starsim as ss, numpy as np, sciris as sc, inspect

d = ss.Disease()
print('ss.Disease() with no args: graph =', d.graph, '| states:', list(d.state_dict.keys()), '| pars:', dict(d.pars))
sir3 = ss.SIR()
print('ss.SIR pars:', dict(sir3.pars))
print('ss.SIR init_prev type:', type(sir3.pars.init_prev).__name__)

S, I, R = ss.states('S I R')
dsl = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), init_prev=0.01, name='sir')
print('declared Disease init_prev:', dsl.graph.init_prev, '| in pars?', 'init_prev' in dsl.pars)
print('declared sir.S type:', type(dsl.S).__name__, '| ss.SIR().susceptible type:', type(sir3.susceptible).__name__)
print('declared sir.state_dict (Starsim-3 Arr states):', list(dsl.state_dict.keys()))
sim = ss.Sim(dsl, n_agents=500, unit='day', stop=5, verbose=0).run()
print('after agent run, state_dict:', list(sim.diseases.sir.state_dict.keys()))
print('sir.S is still a ss.State?', type(sim.diseases.sir.S).__name__, '| engine:', type(sim.diseases.sir.engine).__name__)
print('results:', list(sim.results.sir.keys()))

# Methods a declared disease inherits but which mean nothing for it
methods = [m for m in dir(ss.Disease) if not m.startswith('_')]
print('\npublic attrs on ss.Disease:', methods)
print('\nInfection-only attrs (not in Disease):', sorted(set(dir(ss.Infection)) - set(dir(ss.Disease))))

# "graph is None" branches in diseases.py
src = inspect.getsource(ss.diseases)
print('\n"graph is None"/"graph is not None" branches in diseases.py:', src.count('graph is None') + src.count('graph is not None'))

# Mixing: a declared Disease given to a sim together with an ss.SIR
sim = ss.Sim([ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='a'), ss.SIR(name='b')], n_agents=500, networks=ss.RandomNet(), unit='day', stop=5, verbose=0).run()
print('\nmixed sim ran; results a:', list(sim.results.a.keys())[:6], 'b:', list(sim.results.b.keys())[:6])

# Class form with step()
class SIRplus(ss.Disease):
    S, I, R = ss.states(3)
    infection = S >> I @ 0.3*I
    recovery = I >> R @ ss.days(10)
    def step(self):
        super().step()
        self.n_steps = getattr(self, 'n_steps', 0) + 1
m = SIRplus(init_prev=0.01)
sim = ss.Sim(m, unit='day', stop=5, verbose=1).run()
print('class with step(): solver', sim.pars.solver, 'n_steps', getattr(sim.diseases[m.name], 'n_steps', None))
print('describe:'); sim.diseases[m.name].describe()
