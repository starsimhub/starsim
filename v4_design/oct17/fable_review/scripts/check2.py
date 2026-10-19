import warnings; warnings.simplefilter('ignore')
import io, contextlib, starsim as ss
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10))
    sim = ss.Sim(sir, stop=100, verbose=0).run()
print('2. two-line SIR max n_I', sim.results.sir.n_I.values.max(), 'init I', sim.results.sir.n_I.values[0])
