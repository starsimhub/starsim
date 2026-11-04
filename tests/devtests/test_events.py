"""
Test that agents by events (solver='abm-events') agree with agents by steps at a small step, with the exact compartmental solver on a
well-mixed model, and with epipack's continuous-time runs of the SIARX with contact tracing on a network
"""

# %% Imports and settings
import sys
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl

from paths import lfd

kw = dict(verbose=0)
siarx_dir = lfd/'docs'/'models'/'epipack'/'siarx_tracing'


def summary(msim, name, state):
    """ The mean peak, and the mean final number, of a state over the runs, with the standard error of each """
    vals = np.array([sim.results[name][state].values for sim in msim.sims])
    peak, final = vals.max(axis=1), vals[:, -1]
    se = lambda x: x.std()/np.sqrt(len(x))
    return peak.mean(), se(peak), final.mean(), se(final)


def agree(a, b, tol, what):
    """ Two means agree within a share `tol` of the first, or within three standard errors of their difference """
    diff = abs(a[0] - b[0])
    assert diff < max(tol*a[0], 3*np.hypot(a[1], b[1])), f'{what}: {a[0]:.1f} against {b[0]:.1f}'
    return


def test_events_vs_abm():
    sc.heading('By events at a coarse step, against agents by steps at a fine one: an SEIR with a lognormal latent period on a static network')
    S, E, I, R = ss.states('S E I R')
    seir = ss.Disease(infection=S >> E @ 0.1*I, latent=ss.Transition(E, I, dur=ss.lognorm_ex(mean=ss.days(4), std=ss.days(2))), recovery=I >> R @ 0.2, init_prev=0.01, name='seir')
    base = ss.Sim(seir, networks=ssl.StaticNet(n_contacts=6), n_agents=2000, unit='day', stop=80, **kw)
    ev = ss.parallel(base.new(solver='abm-events', dt=1), n_runs=100, n_cpus=2)
    ab = ss.parallel(base.new(solver='abm', dt=0.05), n_runs=100, n_cpus=2)
    e, a = summary(ev, 'seir', 'I'), summary(ab, 'seir', 'I')
    ef, af = summary(ev, 'seir', 'R')[2:], summary(ab, 'seir', 'R')[2:]
    print(f'Peak of I: {e[0]:.1f} by events, {a[0]:.1f} by steps; final R: {ef[0]:.1f} and {af[0]:.1f}')
    agree(a[:2], e[:2], 0.05, 'peak of I') # The step of 0.05 days leaves agents within about 1% (spec 12)
    agree(af, ef, 0.03, 'final size')
    return


def test_events_vs_gillespie():
    sc.heading('By events in a mixing pool, against the exact compartmental solver')
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=S >> I @ 0.03*I, recovery=ss.Transition(I, R, dur=ss.days(10)), init_prev=0.01, name='sir')
    base = ss.Sim(sir, networks=ss.MixingPool(n_contacts=10), unit='day', stop=100, dt=1, **kw)
    g = ss.parallel(base.new(solver='gillespie', total_pop=2000), n_runs=300, n_cpus=2)
    e = ss.parallel(base.new(solver='abm-events', n_agents=2000), n_runs=300, n_cpus=2)
    for state, i in [('I', 0), ('R', 2)]:
        gs, es = summary(g, 'sir', state), summary(e, 'sir', state)
        print(f'{state}: {gs[i]:.1f} by gillespie, {es[i]:.1f} by events')
        agree(gs[i:i+2], es[i:i+2], 0.02, f'{"peak" if i == 0 else "final"} {state}')
    return


@pytest.mark.skipif(not siarx_dir.exists(), reason='the lingua_franca_design repo is not beside this one (or STARSIM_LFD)')
def test_siarx_tracing():
    sc.heading('epipack\'s SIARX with contact tracing on a network, by events, against epipack\'s mean of 100 runs in continuous time')
    sys.path.insert(0, str(siarx_dir.parent.parent))
    sys.path.insert(0, str(siarx_dir))
    import tools
    from model import sim
    ref = {k:np.array(v[1]) for k,v in tools.read_results(siarx_dir).items()}
    res = ss.parallel(sim.new(solver='abm-events', dt=0.1), n_runs=40, n_cpus=2).results.siarx
    diff = max(np.abs(res[c].values.mean(axis=0)[::10] - ref[f'network_{c}']).max() for c in 'SIARX')/10_000
    print(f'Largest difference: {diff:.2e} of the population') # 5e-3 measured; 'abm' at the same step is 1.2e-2 from it over 100 runs
    assert diff < 1.5e-2
    return


if __name__ == '__main__':
    T = sc.timer()
    test_events_vs_abm()
    test_events_vs_gillespie()
    test_siarx_tracing()
    T.toc()
