"""MultiSim surface: n_runs, results, summarize, extinction, plot"""
import numpy as np
import sciris as sc
import starsim as ss

def make(beta=0.3):
    return ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', dur=10), init=dict(I=3), name='sir')

kw = dict(total_pop=300, unit='day', stop=40, verbose=0, solver='gillespie')
msim = ss.Sim(make(), **kw).run(n_runs=5)
print('type', type(msim).__name__, 'len', len(msim), 'which', msim.which)
print('msim.results type', type(msim.results).__name__, 'keys', list(msim.results.keys()))
nI = msim.results.sir.n_I
print('n_I type', type(nI).__name__, 'values', nI.values.shape, 'quantiles', nI.quantiles([0.5]).shape)
print('select(run=0)', type(nI.select(run=0)).__name__)
print('has to_df?', hasattr(nI, 'to_df'), 'has resample?', hasattr(nI, 'resample'), 'has values', hasattr(nI, 'values'), 'isinstance Result?', isinstance(nI, ss.Result))
try:
    print('nI + 1:', (nI + 1))
except Exception as e:
    print('nI+1 ERR', type(e).__name__, e)
try:
    print('nI[0]', nI[0])
except Exception as e:
    print('nI[0] ERR', type(e).__name__, e)
print('--- summarize')
s = msim.summarize()
print(type(s).__name__, list(s.keys())[:12])
print(s[list(s.keys())[0]])
print('--- extinction')
msim.extinction()
print(msim.extinction(output=True))
print('--- plot')
fig = msim.plot()
print('msim.plot (unreduced) axes', len(fig.axes), 'lines in axis 0', len(fig.axes[0].lines))
try:
    fig = msim.plot(lines=True)
    print('msim.plot(lines=True) ok')
except Exception as e:
    print('msim.plot(lines=True) ERR', type(e).__name__, str(e)[:200])
fig = nI.plot(); print('MultiResult.plot ok')
fig = nI.plot(lines=True); print('MultiResult.plot(lines=True) ok, lines', len(fig.lines))
msim.reduce()
print('after reduce: which', msim.which, 'results type', type(msim.results).__name__, 'keys', list(msim.results.keys())[:8])
print('after reduce msim.results.sir type', type(msim.results.sir).__name__, 'and sir_n_I?', 'sir_n_I' in msim.results)
fig = msim.plot(); print('reduced plot axes', len(fig.axes))
print('msim.results.sir_n_I low/high set?', msim.results.sir_n_I.low is not None)
print('msim.base_sim.results.sir.n_I.low set?', msim.base_sim.results.sir.n_I.low is not None)
# to_df
try:
    df = msim.results.to_df(); print('msim.results.to_df cols', list(df.columns)[:6])
except Exception as e:
    print('msim to_df ERR', type(e).__name__, str(e)[:200])
# scenarios
print('--- scenarios')
ms = ss.parallel(baseline=ss.Sim(make(0.3), **kw), high=ss.Sim(make(0.5), **kw), n_runs=3)
print('labels', ms.results.sir.n_I.labels)
print('select(scenario=high).values', ms.results.sir.n_I.select(scenario='high').values.shape)
d = ms.diff('baseline'); print('diff keys', list(d.keys()), list(d.high.keys()), type(d.high.sir.n_I).__name__)
ms.extinction()
fig = ms.plot(); print('scenario plot ok')
# compare / plot_compare (Starsim 3 surface)
print(ms.compare(output=True).head(6))
# pars sweep
ms2 = ss.parallel(ss.Sim(make(), **kw), pars=dict(beta=[0.2, 0.4]), n_runs=2)
print('run_pars', ms2.run_pars.to_dict('records'), 'labels', ms2.results.sir.n_I.labels, 'n values', ms2.results.sir.n_I.values.shape)
# A Starsim-3 (code) module msim: what's the result surface
m3 = ss.MultiSim(ss.Sim(diseases='sis', networks='random', n_agents=300, start=2000, stop=2002, verbose=0), n_runs=3).run()
print('ss3 msim.results', m3.results, 'type', type(m3.results).__name__)
try: m3.extinction()
except Exception as e: print('ss3 extinction ERR', type(e).__name__, str(e)[:120])
# MultiSim(sim, n_runs=) with a declared sim
m4 = ss.MultiSim(ss.Sim(make(), **kw), n_runs=3).run()
print('MultiSim(...).run() results type', type(m4.results).__name__, m4.results)
