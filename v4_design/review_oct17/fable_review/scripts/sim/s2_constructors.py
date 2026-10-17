import starsim as ss, sciris as sc, numpy as np
def mk(): return ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
def try_(label, fn):
    sc.heading(label)
    try:
        s = fn()
        print(repr(s))
        return s
    except Exception as E:
        print('ERROR', type(E).__name__, str(E)[:300])

try_('1 positional module', lambda: ss.Sim(mk(), stop=10, verbose=0).run())
try_('2 positional list', lambda: ss.Sim([mk(), ss.RandomNet()], stop=10, verbose=0).run())
try_('3 diseases=', lambda: ss.Sim(diseases=mk(), stop=10, verbose=0).run())
try_('4 modules=', lambda: ss.Sim(modules=[mk()], stop=10, verbose=0).run())
try_('5 dict first arg', lambda: ss.Sim(dict(diseases='sir', networks='random', n_agents=500, stop=10, verbose=0)).run())
try_('6 pars= dict of sim pars', lambda: ss.Sim(pars=dict(diseases='sir', networks='random', n_agents=500, stop=10, verbose=0)).run())
try_('7 pars= model values', lambda: ss.Sim(mk(), pars=dict(beta=0.1), stop=10, verbose=0).run())
s = try_('8 pars= mixed model + sim', lambda: ss.Sim(mk(), pars=dict(beta=0.1, verbose=0), stop=10).run())
try_('9 custom=', lambda: ss.Sim(custom=mk(), stop=10, verbose=0).run())
try_('10 string disease, no network', lambda: ss.Sim(diseases='sir', n_agents=500, stop=10, verbose=0).run())
try_('11 nothing', lambda: ss.Sim(stop=10, verbose=0).run())
try_('12 demographics=True with declared', lambda: ss.Sim(mk(), demographics=True, stop=10, verbose=0).run())
try_('13 birth_rate= with declared (ode)', lambda: ss.Sim(mk(), birth_rate=20, death_rate=10, stop=10, verbose=0).run())
try_('14 birth_rate= with declared (abm)', lambda: ss.Sim(mk(), birth_rate=20, death_rate=10, n_agents=500, stop=10, verbose=0).run())
try_('15 Disease subclass positional, name from variable?', lambda: None)
class MySIR(ss.Disease):
    S, I, R = ss.states(3)
    infection = S >> I @ 0.05 * I
    recovery = I >> R @ 10
flu = MySIR()
s = ss.Sim(flu, stop=10, verbose=0).run()
print('subclass name:', s.diseases.keys(), ' results:', list(s.results.keys()))
measles = mk()
s = ss.Sim(measles, stop=10, verbose=0).run()
print('plain Disease name from variable:', s.diseases.keys())
sc.heading('16 default stop for declared vs coded')
s = ss.Sim(mk(), verbose=0).run(); print('declared default:', s.t.start, s.t.stop, s.t.dt, s.t.npts)
s = ss.Sim(ss.SIR(), networks=ss.RandomNet(), n_agents=200, verbose=0).run(); print('coded default:', s.t.start, s.t.stop, s.t.dt, s.t.npts)
sc.heading('17 Sim public methods')
meths = [m for m in dir(ss.Sim) if not m.startswith('_') and callable(getattr(ss.Sim, m, None))]
print(len(meths), meths)
props = [m for m in dir(ss.Sim) if not m.startswith('_') and isinstance(getattr(ss.Sim, m, None), property)]
print('props', props)
