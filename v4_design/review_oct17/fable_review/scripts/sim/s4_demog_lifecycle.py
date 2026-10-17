import starsim as ss, sciris as sc, numpy as np
def try_(label, fn):
    sc.heading(label)
    try:
        return fn()
    except Exception as E:
        print('ERROR', type(E).__name__, str(E)[:400])

sc.heading('A: declared births/deaths vs Births/Deaths modules')
def mk():
    return ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.3)), recovery=ss.progression('I -> R', dur=ss.days(10)),
                      births=ss.birth('-> S', rate=ss.peryear(0.02)), deaths=ss.death(rate=ss.peryear(0.01)), init_prev=0.01, name='sir')
s_ode = ss.Sim(mk(), start='2020-01-01', stop='2021-01-01', verbose=0).run()
print('declared ode: solver', s_ode.solver, 'results', list(s_ode.results.sir.keys()), 'people results', [k for k in s_ode.results.keys() if k!='sir'])
N0 = s_ode.results.sir.n_S[0]+s_ode.results.sir.n_I[0]+s_ode.results.sir.n_R[0]; N1 = s_ode.results.sir.n_S[-1]+s_ode.results.sir.n_I[-1]+s_ode.results.sir.n_R[-1]
print(f'  pop {N0:,.0f} -> {N1:,.0f}; births total {s_ode.results.sir.births.sum():,.1f}; deaths total {s_ode.results.sir.deaths.sum():,.1f}')
s_abm = ss.Sim(mk(), n_agents=2000, start='2020-01-01', stop='2021-01-01', verbose=0).run()
print('declared abm: results', list(s_abm.results.sir.keys()), 'people results', [k for k in s_abm.results.keys() if k!='sir'])
print(f'  n_alive {s_abm.results.n_alive[0]} -> {s_abm.results.n_alive[-1]}; births {s_abm.results.sir.births.sum()}; deaths(sir) {s_abm.results.sir.deaths.sum()}; new_deaths(people) {s_abm.results.new_deaths.sum()}; use_aging {s_abm.pars.use_aging}')
print('  loop labels:', sorted(set(s_abm.loop.to_df().label)))

sir2 = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.3)), recovery=ss.progression('I -> R', dur=ss.days(10)), init_prev=0.01, name='sir')
s_mod = ss.Sim(sir2, demographics=[ss.Births(birth_rate=ss.peryear(20)), ss.Deaths(death_rate=ss.peryear(10))], n_agents=2000, start='2020-01-01', stop='2021-01-01', verbose=0).run()
print('modules abm: results', list(s_mod.results.keys()), list(s_mod.results.births.keys()), list(s_mod.results.deaths.keys()))
print(f'  n_alive {s_mod.results.n_alive[0]} -> {s_mod.results.n_alive[-1]}; births {s_mod.results.births.new.sum()}; deaths {s_mod.results.deaths.new.sum()}; use_aging {s_mod.pars.use_aging}')
print('  newborns S?', s_mod.diseases.sir.S.sum() if hasattr(s_mod.diseases.sir,'S') else None)
print('  loop labels:', sorted(set(s_mod.loop.to_df().label)))
try_('modules with ODE (solver=ode forced)', lambda: ss.Sim(sir2, demographics=[ss.Births(), ss.Deaths()], solver='ode', start='2020-01-01', stop='2021-01-01', verbose=0).run())
sir3 = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.3)), recovery=ss.progression('I -> R', dur=ss.days(10)), init_prev=0.01, name='sir')
s = ss.Sim(sir3, demographics=[ss.Births(), ss.Deaths()], start='2020-01-01', stop='2021-01-01', verbose=1)
s.init(); print('auto with Births ->', s.solver, s.solver_reason)

sc.heading('B: lifecycle and loop for declared vs coded')
s_c = ss.Sim(ss.SIR(), networks=ss.RandomNet(), n_agents=500, stop=5, verbose=0).run()
print('coded loop labels:', sorted(set(s_c.loop.to_df().label)))
s_d = ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir'), n_agents=500, stop=5, verbose=0).run()
print('declared abm loop labels:', sorted(set(s_d.loop.to_df().label)))
s_o = ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir'), stop=5, verbose=0).run()
print('declared ode loop:', s_o.loop.plan is not None and len(s_o.loop.plan), 'kernel_only', s_o.kernel_only())
s_o2 = ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir'), stop=5, verbose=0)
s_o2.init(); s_o2.loop.insert(lambda sim: None, label='sir.step'); s_o2.run()
print('declared ode loop (with insert) labels:', sorted(set(s_o2.loop.to_df().label)))

sc.heading('C: class form with step() override forces abm?')
class SIRx(ss.Disease):
    S, I, R = ss.states(3)
    infection = S >> I @ 0.05 * I
    recovery = I >> R @ ss.days(10)
    def step(self):
        super().step()
        self.n_custom = 1
s = ss.Sim(SIRx(init_prev=0.01), stop=5, verbose=1); s.init(); print('->', s.solver, s.solver_reason)
class SIRy(ss.Disease):
    S, I, R = ss.states(3)
    infection = S >> I @ 0.05 * I
    recovery = I >> R @ ss.days(10)
    def step(self):
        super().step()
    compartmental = True
s = ss.Sim(SIRy(init_prev=0.01), stop=5, verbose=1); s.init(); print('with compartmental=True ->', s.solver, s.solver_reason)
def f(sim): pass
s = ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, extra=f), stop=5, verbose=1); s.init(); print('plain function member ->', s.solver, s.solver_reason)
