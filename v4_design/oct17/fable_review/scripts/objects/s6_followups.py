"""Follow-ups: pars (fixed names), sim.unit for Starsim-3 sims, stop=2000, typed legacy pars in a unitless sim, naming of bare variables, spec-17 method existence"""
import os, warnings
os.environ['MPLBACKEND'] = 'agg'
import numpy as np
import starsim as ss
ss.options.verbose = 0

def mk(beta=0.05):
    return ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', dur=10), name='sir')

def final_I(sim):
    sim.run()
    return float(sim.results.sir.n_I[-1])

print('=== A. ways to set beta ===')
base = final_I(ss.Sim(mk(), stop=50))
a = mk(); a.pars.beta = 0.1
b = mk(); b.pars.beta.value = 0.1
c = mk(); c.set_pars(beta=0.1)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    e = mk(); e.pars.update(beta=0.1)
    print('  pars.update warns:', [str(x.message).strip()[:80] for x in w])
res = dict(base=base, attr_assign=final_I(ss.Sim(a, stop=50)), value_mutate=final_I(ss.Sim(b, stop=50)), set_pars=final_I(ss.Sim(c, stop=50)),
           pars_update=final_I(ss.Sim(e, stop=50)), sim_pars=final_I(ss.Sim(mk(), pars=dict(beta=0.1), stop=50)),
           sim_pars_dotted=final_I(ss.Sim(mk(), pars={'sir.beta': 0.1}, stop=50)), ctor=final_I(ss.Sim(mk(0.1), stop=50)))
for k, v in res.items():
    print(f'  {k:18s} n_I[-1] = {v:.2f}')
print('  after attr_assign a.pars.beta =', repr(a.pars.beta), type(a.pars.beta).__name__)
d = ss.Disease(infection=ss.transmission('S -> I', beta=ss.par(0.05, bounds=[0.01, 0.5])), recovery=ss.progression('I -> R', dur=10), name='sir')
d.pars.beta = 0.2
print('  bounds after sir.pars.beta = 0.2:', getattr(d.pars.beta, 'bounds', 'LOST (now a float)'))
try:
    f = final_I(ss.Sim(pars=dict(diseases=mk(), stop=50)))
    print('  Starsim-3 dict pars first arg: ok', round(f, 2))
except Exception as E:
    print('  Starsim-3 dict pars ERROR', type(E).__name__, str(E)[:200])
d = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.05)), recovery=ss.progression('I -> R', dur=ss.days(10)), name='sir')
d.pars.beta = 0.1
print('  typed par after sir.pars.beta = 0.1:', repr(d.pars.beta), '(unit lost silently?)')
s3 = ss.SIR(beta=0.05)
print('  ss.SIR pars types:', {k: type(v).__name__ for k, v in s3.pars.items()})
beta = ss.par(0.05)
S, I, R = ss.states(3)
d1 = ss.Disease(S >> I @ beta * I, I >> R @ 10, name='d1')
S2, I2, R2 = ss.states('S2 I2 R2')
d2 = ss.Disease(S2 >> I2 @ beta * I2, I2 >> R2 @ 10, name='d2')
print('  shared par: same object?', d1.pars.beta is d2.pars.beta)
sim = ss.Sim([d1, d2], pars=dict(beta=0.1), stop=5)
try:
    sim.init(); print('  pars=dict(beta=0.1) with two users ->', [m.pars.beta.value for m in sim.diseases.values()])
except Exception as E:
    print('  shared beta via pars= ERROR', type(E).__name__, str(E)[:250])
sim = ss.Sim(mk(), stop=5).init()
try:
    sim.diseases.sir.pars.newkey = 3; print('  add new key to locked Pars after init: accepted')
except Exception as E:
    print('  add new key:', type(E).__name__)

print('=== B. a Starsim-3 sim: two notions of unit ===')
sim = ss.Sim(ss.SIR(), stop=10).init()
print('  sim.unit =', sim.unit, '| sim.pars.unit =', sim.pars.unit, '| sim.t.dt =', repr(sim.t.dt), '| sim.pars.start =', sim.pars.start, '| sim.pars.stop =', sim.pars.stop)
sim = ss.Sim(mk(), stop=10).init()
print('  declared: sim.unit =', sim.unit, '| sim.pars.unit =', sim.pars.unit, '| sim.t.dt =', repr(sim.t.dt))

print('=== C. a typed Starsim-3 module inside a unitless declared sim ===')
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    sim = ss.Sim([ss.SIS(), mk()], stop=100).init()
    sis = sim.diseases.sis
    print('  sim.unit =', sim.unit, '| sim.t.dt =', repr(sim.t.dt), '| sis.pars.dur_inf =', sis.pars.dur_inf, '| in steps:', sis.pars.dur_inf.to_dt() if hasattr(sis.pars.dur_inf, 'to_dt') else '?', '| sis.pars.beta', sis.pars.beta, '-> per step', sis.pars.beta.to_dt() if hasattr(sis.pars.beta, 'to_dt') else '?')
    print('  warnings:', [str(x.message).strip()[:100] for x in w])

print('=== D. stop that looks like a year ===')
for kw in [dict(stop=2000), dict(stop=1950), dict(start=0, stop=2000), dict(stop=1899), dict(start=2000, stop=2050, dt=ss.days(1))]:
    sim = ss.Sim(mk(), **kw).init()
    print(f'  {kw!s:45s} unit={sim.unit!r} source={sim.unit_source!r} npts={len(sim.t)} start={sim.t.start!r} stop={sim.t.stop!r}')

print('=== E. names of bare variables in slots ===')
S, I, R = ss.states('S I R')
beta = 0.05; gamma = 0.1; contacts = ss.timeseries({0: 1.0, 30: 0.5}); dur_inf = ss.days(10)
d = ss.Disease(S >> I @ beta * contacts * I, I >> R @ dur_inf, name='x')
print('  beta=0.05; gamma=0.1; contacts=ss.timeseries; dur_inf=ss.days(10) ->', list(d.pars.keys()))
S, I, R = ss.states('S I R')
d = ss.Disease(S >> I @ beta * I, I >> R @ gamma, name='x')
print('  rate from variable gamma ->', list(d.pars.keys()))
S, I, R = ss.states('S I R')
gamma = ss.par(0.1)
d = ss.Disease(S >> I @ beta * I, I >> R @ gamma, name='x')
print('  gamma as ss.par ->', list(d.pars.keys()))

print('=== F. describe(verbose=True) and W102 for @ 10 ===')
ss.options.verbose = 1
S, I, R = ss.states('S I R')
d = ss.Disease(S >> I @ 0.05 * I, I >> R @ 10, name='x')
sim = ss.Sim(d, stop=20)
try:
    sim.describe(verbose=True)
except TypeError as E:
    print('  describe(verbose=True):', E)
    sim.init()
ss.options.verbose = 0

print('=== G. names the spec (17) promises, checked with hasattr ===')
sim = ss.Sim(mk(), stop=5).init()
for obj, label, names in [(ss, 'ss', ['func', 'data', 'people', 'contacts', 'geo', 'gravity', 'Mobility', 'PartnerNet', 'compare_models', 'diff', 'Requires', 'Vaccine', 'Test', 'Treatment', 'vaccinate', 'screen', 'treat', 'change', 'modify', 'erlang', 'null', 'stratify', 'par', 'outcome', 'states', 'transfer', 'birth', 'death', 'load', 'save', 'options']),
                          (sim, 'sim', ['init', 'run', 'plot', 'summarize', 'to_df', 'loop', 'results', 'describe', 'solvers', 'stratify', 'requires', 'to_json', 'hash', 'calibrate', 'sensitivity', 'check_dt', 'what_modifies', 'crn_report', 'to_spec', 'equations', 'diagram', 'R0']),
                          (sim.diseases.sir, 'sir', ['describe', 'diagram', 'equations', 'solvers', 'transitions', 'stratify', 'to_json', 'copy', 'pars', 'R0', 'to_spec']),
                          (sim.diseases.sir.S, 'state', ['n', 'prev', 'frac']),
                          (sim.results.sir.n_I, 'result', ['select', 'by', 'sum', 'resample', 'quantiles', 'values']),
                          (sim.loop, 'loop', ['insert', 'to_df', 'plot', 'plot_step_order'])]:
    missing = [n for n in names if not hasattr(obj, n)]
    print(f'  {label}: missing {missing}')
print('  ss.Expr exported (spec 17 says removed):', hasattr(ss, 'Expr'), '| ss.seasonality and ss.seasonal both exist:', hasattr(ss, 'seasonality'), hasattr(ss, 'seasonal'), type(ss.seasonality), type(ss.seasonal))
print('  ss.immigration/emigration exported (not in 17):', hasattr(ss, 'immigration'), hasattr(ss, 'emigration'))
