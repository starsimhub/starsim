"""
Test that agents by events (solver=ss.abm(step='events')) agree with agents by steps at a small step (including the infection log, networks
with acts, triggers, and stratified models), with the exact compartmental solver on a well-mixed model, and with epipack's
continuous-time runs of the SIARX with contact tracing on a network
"""

# %% Imports and settings
import sys
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl

from paths import lf

kw = dict(verbose=0)
siarx_dir = lf/'docs'/'models'/'epipack'/'siarx_tracing'


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
    seir = ss.Disease(infection=S >> E @ 0.1*I, latent=ss.Transition(E, I, dur=ss.lognormal(mean=ss.days(4), std=ss.days(2))), recovery=I >> R @ 0.2, init_prev=0.01, name='seir')
    base = ss.Sim(seir, networks=ssl.StaticNet(n_contacts=6), n_agents=2000, unit='day', stop=80, **kw)
    ev = ss.parallel(base.new(solver=ss.abm(step='events'), dt=1), n_runs=100, workers=2)
    ab = ss.parallel(base.new(solver='abm', dt=0.05), n_runs=100, workers=2)
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
    g = ss.parallel(base.new(solver='gillespie', total_pop=2000), n_runs=300, workers=2)
    e = ss.parallel(base.new(solver=ss.abm(step='events'), n_agents=2000), n_runs=300, workers=2)
    for state, i in [('I', 0), ('R', 2)]:
        gs, es = summary(g, 'sir', state), summary(e, 'sir', state)
        print(f'{state}: {gs[i]:.1f} by gillespie, {es[i]:.1f} by events')
        agree(gs[i:i+2], es[i:i+2], 0.02, f'{"peak" if i == 0 else "final"} {state}')
    return


def test_events_infection_log():
    sc.heading('The infection log by events, against agents by steps: who infected whom, and over which route')
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=S >> I @ 0.04*I, recovery=I >> R @ 0.2, init_prev=0.01, name='sir')
    base = ss.Sim(sir, networks=[ssl.StaticNet(n_contacts=6), ss.MixingPool(n_contacts=3)], n_agents=2000, unit='day', stop=60, analyzers='infection_log', **kw)
    ev = ss.parallel(base.new(solver=ss.abm(step='events'), dt=1), n_runs=40, workers=2)
    ab = ss.parallel(base.new(solver='abm', dt=0.1), n_runs=40, workers=2)
    def shares(msim):
        """ The share of the infections (not seeds) over the network, and the number logged against the flow """
        out = []
        for sim in msim.sims:
            df = sim.analyzers[0].to_df()
            seeds = (df.source == -1).fillna(False) # By steps, an infection in a pool has no source (NA)
            inf = df[~seeds]
            assert len(inf) == sim.results.sir.infection.values.sum() and seeds.sum() == sim.results.sir.I[0] # Each infection once, and the seeds
            out.append((inf.network == 'staticnet').mean())
        return np.mean(out), np.std(out)/np.sqrt(len(out))
    for sim in ev.sims[:3]: # By events, the source of an infection in a pool is known too, and was infected before it
        df = sim.analyzers[0].to_df()
        inf = df[df.source != -1]
        assert inf.network.isin(['staticnet', 'mixingpool']).all()
        first = df.groupby('target').t.min()
        assert np.all(first[inf.source.astype(int)].values <= inf.t.values)
    e, a = shares(ev), shares(ab)
    print(f'Share of infections over the network: {e[0]:.3f} by events, {a[0]:.3f} by steps')
    agree(a, e, 0.03, 'share over the network')
    return


def test_events_acts():
    sc.heading('A network with acts by events, against agents by steps: a probability per act in each direction, on partnerships that change')
    def make(**kwargs):
        S, I = ss.states('S I')
        sti = ss.Disease(infection=S >> I @ dict(sexualnet=ss.peract(m2f=0.08, f2m=0.04))*I, recovery=I >> S @ ss.peryear(0.2), init_prev=0.1, name='sti')
        return ss.Sim(sti, networks=ss.SexualNet(acts=ss.constant(ss.per(80, 'year'))), n_agents=3000, start=2000, stop=2015, **kwargs, **kw)
    ev = ss.parallel(make(solver=ss.abm(step='events'), dt=1/12), n_runs=40, workers=2)
    ab = ss.parallel(make(solver='abm', dt=1/52), n_runs=40, workers=2)
    e, a = summary(ev, 'sti', 'I'), summary(ab, 'sti', 'I')
    print(f'Final prevalence: {e[2]:.1f} by events, {a[2]:.1f} by steps')
    agree(a[2:], e[2:], 0.05, 'final number infected')
    return


def test_events_triggers():
    sc.heading('Triggers (ss.on) by events, at the end of the step of their event, against agents by steps: isolation two days after onset')
    S, E, I, Q, R = ss.states('S E I Q R')
    d = ss.Disease(infection=S >> E @ 0.3*I, onset=E >> I @ 0.5, recovery=I >> R @ 0.2, isolate=ss.Transition(I, Q, when=ss.on(I.enter, delay=ss.constant(ss.days(2)))),
                   release=Q >> R @ 0.2, init_prev=0.01, name='d')
    base = ss.Sim(d, networks=ssl.StaticNet(n_contacts=8), n_agents=3000, unit='day', stop=80, **kw)
    ev = ss.parallel(base.new(solver=ss.abm(step='events'), dt=0.25), n_runs=40, workers=2)
    ab = ss.parallel(base.new(solver='abm', dt=0.05), n_runs=40, workers=2)
    for state in ['Q', 'R']:
        e, a = summary(ev, 'd', state), summary(ab, 'd', state)
        i = 0 if state == 'Q' else 2
        print(f'{"Peak" if i == 0 else "Final"} {state}: {e[i]:.1f} by events, {a[i]:.1f} by steps')
        agree(a[i:i+2], e[i:i+2], 0.04, f'{"peak" if i == 0 else "final"} {state}') # Each isolation up to a step late, at a quarter of a day
    return


def test_events_groups():
    sc.heading('Stratified models by events, against agents by steps: infections by group, by a contact matrix by age, by risk, and within places')
    S, I, R = ss.states('S I R')
    measles = ss.Disease(S >> I @ 0.14*I, I >> R @ ss.days(8), init=dict(R=ss.by(age=[0.8, 0.92, 0.98])), init_n=dict(I=10), name='m')
    measles.stratify(age={0: 0.3, 15: 0.6, 65: 0.1})
    by_age = ss.Sim(measles, networks=ss.contacts(ages=[0, 15, 65], matrix=[[10, 6, 1], [3, 8, 1], [3, 6, 3]]), n_agents=5000, unit='day', stop=180, **kw)
    S, I, R = ss.states('S I R')
    m = ss.Disease(S >> I @ 0.1*I, I >> R @ ss.days(6), init_prev=0.005, name='m')
    m.stratify(risk={'low': 0.8, 'high': 0.2})
    by_risk = ss.Sim(m, connectors=ss.modify(m.rel_sus, multiply=ss.by(risk=[1, 2])), networks=ss.contacts(risk=[[3, 1], [1, 8]]), n_agents=4000, unit='day', stop=80, **kw)
    S, I, R = ss.states('S I R')
    m = ss.Disease(S >> I @ 0.03*I, I >> R @ ss.days(6), init_prev=0.01, name="m")
    m.stratify(age={0: 0.3, 15: 0.6, 65: 0.1}, patch=ss.geo.geography(names=['a', 'b'], populations=[6000, 4000], x=[0, 20], y=[0, 0])) # Places with no movement between them
    in_places = ss.Sim(m, networks=ss.contacts(ages=[0, 15, 65], matrix=[[10, 6, 1], [3, 8, 1], [3, 6, 3]]), n_agents=4000, unit='day', stop=100, **kw)
    for name, base, dim, levels in [('by age', by_age, 'age', ['0-14', '15-64', '65+']), ('by risk', by_risk, 'risk', ['low', 'high']), ('within places', in_places, 'age', ['0-14', '15-64', '65+'])]:
        ev = ss.parallel(base.new(solver=ss.abm(step='events'), dt=1), n_runs=30, workers=2)
        ab = ss.parallel(base.new(solver='abm', dt=0.1), n_runs=30, workers=2)
        for level in levels:
            got = [np.array([sim.results.m.infection.select(**{dim: level}).sum() for sim in msim.sims]) for msim in [ab, ev]]
            a, e = [(x.mean(), x.std()/np.sqrt(len(x))) for x in got]
            print(f'{name}, {level}: {e[0]:.1f} infections by events, {a[0]:.1f} by steps')
            agree(a, e, 0.05, f'infections {name}, {level}')
    return


@pytest.mark.skipif(not siarx_dir.exists(), reason='the lingua_franca repo is not beside this one (or STARSIM_LF)')
def test_siarx_tracing():
    sc.heading('epipack\'s SIARX with contact tracing on a network, by events, against epipack\'s mean of 100 runs in continuous time')
    sys.path.insert(0, str(siarx_dir.parent.parent))
    sys.path.insert(0, str(siarx_dir))
    import tools
    from model import sim
    ref = {k:np.array(v[1]) for k,v in tools.read_results(siarx_dir).items()}
    res = ss.parallel(sim.new(solver=ss.abm(step='events'), dt=0.1), n_runs=40, workers=2).results.siarx
    diff = max(np.abs(res[c].values.mean(axis=0)[::10] - ref[f'network_{c}']).max() for c in 'SIARX')/10_000
    print(f'Largest difference: {diff:.2e} of the population') # 5e-3 measured; 'abm' at the same step is 1.2e-2 from it over 100 runs
    assert diff < 1.5e-2
    return


if __name__ == '__main__':
    T = sc.timer()
    test_events_vs_abm()
    test_events_vs_gillespie()
    test_events_infection_log()
    test_events_acts()
    test_events_triggers()
    test_events_groups()
    test_siarx_tracing()
    T.toc()
