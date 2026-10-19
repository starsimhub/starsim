import os; os.environ['MPLBACKEND']='agg'
import starsim as ss, warnings
ss.options.verbose = 0
def mk(beta=0.3):
    return ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')
def fin(sim):
    sim.run(); return float(sim.results.sir.n_R[-1])
a = mk(); a.pars.beta = 0.6
b = mk(); b.pars.beta.value = 0.6
c = mk(); c.set_pars(beta=0.6)
with warnings.catch_warnings():
    warnings.simplefilter('ignore'); e = mk(); e.pars.update(beta=0.6)
res = dict(base03=fin(ss.Sim(mk(), stop=60)), ctor06=fin(ss.Sim(mk(0.6), stop=60)), attr=fin(ss.Sim(a, stop=60)), value=fin(ss.Sim(b, stop=60)), set_pars=fin(ss.Sim(c, stop=60)), update=fin(ss.Sim(e, stop=60)), simpars=fin(ss.Sim(mk(), pars=dict(beta=0.6), stop=60)))
for k,v in res.items(): print(f'{k:10s} n_R[-1]={v:.1f}')
# Does ss.modify on sir.pars.beta still work after attr assignment replaced the par with a float?
a = mk(); a.pars.beta = 0.6
try:
    sim = ss.Sim(a, interventions=ss.modify(a.pars.beta, multiply=0.5, start=10), stop=60); sim.run(); print('modify after attr assign: ran, n_R', float(sim.results.sir.n_R[-1]))
except Exception as E: print('modify after attr assign ERROR:', type(E).__name__, str(E)[:200])
b = mk()
sim = ss.Sim(b, interventions=ss.modify(b.pars.beta, multiply=0.5, start=10), stop=60); sim.run(); print('modify on intact par: n_R', float(sim.results.sir.n_R[-1]))
