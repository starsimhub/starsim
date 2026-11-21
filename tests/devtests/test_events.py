"""
Test events and clocks as agents, and that agents by events (solver=ss.solvers.abm(step='events')) agree with agents by steps at a small step,
and with the exact compartmental solver
"""

# %% Imports and settings
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl

from corpus import BV

kw = dict(verbose=0)


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


def make_seir(**kwargs):
    S, E, I, R, D = ss.states('S E I R D')
    infected = E | I
    return ss.Disease(S, E, I, R, D,
        infection = S >> E @ 0.3*I,
        onset     = E >> I @ ss.days(3),
        outcome   = ss.Transition(I, {R: 0.9, D: None}, dur=7),
        init_prev = 0.01, name='d', **kwargs), infected


@sc.timer()
def test_events():
    sc.heading('Testing events: transitions, branches, and entering and leaving states')
    checked = sc.obj(n=0)
    def check(sim): # After the transmission, every event of the step is visible
        d = sim.diseases.d
        st = d.states
        infected = st.E | st.I
        inf, onset = d.infection.uids, d.onset.uids
        assert np.array_equal(infected.enter.uids, inf) # Entering the set from outside; E -> I is within it
        assert np.array_equal(st.I.enter.uids, onset)
        assert np.array_equal(infected.exit.uids, d.outcome.uids)
        assert np.array_equal(np.sort(ss.uids.concatenate([d.outcome.to(st.R).uids, d.outcome.to(st.D).uids])), np.sort(d.outcome.uids))
        assert d.R[d.outcome.to(st.R).uids].all()
        checked.n += len(inf)
    d, infected = make_seir(check=check)
    sim = ss.Sim(d, n_agents=5000, unit='day', stop=60, **kw).run()
    assert checked.n > 1000

    # As conditions, and in on=: one delivery for each person who enters the set, and each who dies
    for on, flow in [(lambda d: (d.E | d.I).enter, 'infection'), (lambda d: d.outcome.to(d.D), None)]:
        d, infected = make_seir()
        event = on(d)
        vx = ss.vaccinate(ss.Vaccine(efficacy=0), when=event, name='vx')
        res = ss.Sim(d, interventions=vx, n_agents=5000, unit='day', stop=60, **kw).run().results
        target = res.d[flow][:-1].sum() if flow else res.d.D[-1] # Events in a step are acted on by the next, so not those of the last (D[-1] is the state at the last step's start)
        assert np.isclose(res.vx.delivered.sum(), target, rtol=0.02), f'{event}: {res.vx.delivered.sum()} delivered, {target} expected'

    # A branch is an event under a compartmental solver too; entering a state is not, yet
    d, infected = make_seir()
    ode = ss.Sim(d, interventions=ss.vaccinate(ss.Vaccine(efficacy=0), when=d.outcome.to(d.D), name='vx'), solver='ode', total_pop=5000, unit='day', stop=60, **kw).run().results
    assert np.isclose(ode.vx.delivered.sum(), 0.1*ode.d.outcome[:-1].sum(), rtol=0.01)
    d, infected = make_seir()
    with pytest.raises(ss.CapabilityError):
        ss.Sim(d, interventions=ss.vaccinate(ss.Vaccine(efficacy=0), when=infected.enter), solver='ode', **kw).run()
    return sim


@sc.timer()
def test_entry_hazard():
    sc.heading('As agents, people entering a state in the state-change row face half a step of its hazards then')
    kw2 = dict(n_agents=50_000, start=2000, stop=2005, unit='year', **kw)
    weekly, monthly = [ss.Sim(BV(name='bv'), dt=dt, **kw2).run().results.bv.recur.sum() for dt in [ss.weeks(1), ss.months(1)]]
    assert np.isclose(weekly, monthly, rtol=0.03), f'weekly {weekly:n}, monthly {monthly:n}' # Without the hazard on entry, monthly steps were about 20% lower
    return


def test_events_vs_abm():
    sc.heading('By events at a coarse step, against agents by steps at a fine one: an SEIR with a lognormal latent period on a static network')
    S, E, I, R = ss.states('S E I R')
    seir = ss.Disease(infection=S >> E @ 0.1*I, latent=ss.Transition(E, I, dur=ss.lognormal(mean=ss.days(4), std=ss.days(2))), recovery=I >> R @ 0.2, init_prev=0.01, name='seir')
    base = ss.Sim(seir, networks=ssl.StaticNet(n_contacts=6), n_agents=2000, unit='day', stop=80, **kw)
    ev = ss.parallel(base.new(solver=ss.solvers.abm(step='events'), dt=1), n_runs=100, workers=2)
    ab = ss.parallel(base.new(solver='abm', dt=0.05), n_runs=100, workers=2)
    e, a = summary(ev, 'seir', 'I'), summary(ab, 'seir', 'I')
    ef, af = summary(ev, 'seir', 'R')[2:], summary(ab, 'seir', 'R')[2:]
    agree(a[:2], e[:2], 0.05, 'peak of I') # The step of 0.05 days leaves agents within about 1% (spec 12)
    agree(af, ef, 0.03, 'final size')
    return


def test_events_vs_gillespie():
    sc.heading('By events in a mixing pool, against the exact compartmental solver')
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=S >> I @ 0.03*I, recovery=ss.Transition(I, R, dur=ss.days(10)), init_prev=0.01, name='sir')
    base = ss.Sim(sir, networks=ss.MixingPool(n_contacts=10), unit='day', stop=100, dt=1, **kw)
    g = ss.parallel(base.new(solver='gillespie', total_pop=2000), n_runs=300, workers=2)
    e = ss.parallel(base.new(solver=ss.solvers.abm(step='events'), n_agents=2000), n_runs=300, workers=2)
    for state, i in [('I', 0), ('R', 2)]:
        gs, es = summary(g, 'sir', state), summary(e, 'sir', state)
        agree(gs[i:i+2], es[i:i+2], 0.02, f'{"peak" if i == 0 else "final"} {state}')
    return


if __name__ == '__main__':
    T = sc.timer()
    test_events()
    test_entry_hazard()
    test_events_vs_abm()
    test_events_vs_gillespie()
    T.toc()
