import starsim as ss, traceback, numpy as np
print('=== V3 initial split')
S, E, I, R, D = ss.states(5)
infection = S >> E @ 0.3*I
latent = E >> I @ ss.days(5)
outcome = I >> (0.9*R + 0.1*D) @ ss.days(10)
seir = ss.Disease(infection, latent, outcome, init_prev=0.001)
seir.stratify(vaccinated=[False, True], adjust={'infection': [1.0, 0.2]})
sim = ss.Sim(seir, start='2020-01-01', stop='2020-02-01', total_pop=1e6, verbose=0).run()
r = sim.results.seir.n_S
print('n_S at t=0: unvaccinated', r.select(vaccinated=False).values[0], 'vaccinated', r.select(vaccinated=True).values[0])
print('=== verbose default:', ss.options.verbose)
print('=== msim with n_runs: what is plotted?')
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
msim = ss.parallel(ss.Sim(sir, stop=50, solver='stochastic', verbose=0), n_runs=5)
r = msim.results.sir.n_I
print(type(r).__name__, r.values.shape)
for m in ['quantiles', 'median', 'mean', 'summarize']:
    try: out = getattr(r, m)(); print(m, '->', type(out).__name__, getattr(out, 'shape', None) or str(out)[:100])
    except Exception as e: print(m, 'FAIL', type(e).__name__, str(e)[:100])
print([k for k in dir(msim) if not k.startswith('_')][:60])
print('=== describe vs print(sim) vs sim.summarize for the ODE sim')
print(repr(sim))
try: print(sim.summarize())
except Exception as e: print('summarize FAIL', e)
print('=== two-line SIR errors on a flat run?')
sir0 = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10))
sim0 = ss.Sim(sir0, stop=100).run()
print('n_I max', sim0.results.sir0.n_I.values.max())
