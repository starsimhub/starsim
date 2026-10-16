"""
Test events (transitions, branches, entering and leaving states) and clocks (time in a state or a set of states)
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss

kw = dict(verbose=0)


def make_seir(**kwargs):
    S, E, I, R, D = ss.states('S E I R D')
    infected = E | I
    return ss.Disease(S, E, I, R, D,
        infection = S >> E @ 0.3*I,
        onset     = E >> I @ ss.days(3),
        outcome   = ss.Transition(I, [R, D], p=[0.9, None], dur=7),
        init_prev = 0.01, name='d', **kwargs), infected


@sc.timer()
def test_events():
    sc.heading('Testing events: transitions, branches, and entering and leaving states')
    checked = sc.objdict(n=0)
    def check(sim): # After the transmission, every event of the step is visible
        d = sim.diseases.d
        st = d.graph.states
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
        vx = ss.vaccinate(ss.Vaccine(efficacy=0), on=event, name='vx')
        res = ss.Sim(d, interventions=vx, n_agents=5000, unit='day', stop=60, **kw).run().results
        target = res.d[flow][:-1].sum() if flow else res.d.n_D[-2] # Events in a step are acted on by the next, so not those of the last
        assert np.isclose(res.vx.delivered.sum(), target, rtol=0.02), f'{event}: {res.vx.delivered.sum()} delivered, {target} expected'

    # A branch is an event under a compartmental solver too; entering a state is not, yet
    d, infected = make_seir()
    ode = ss.Sim(d, interventions=ss.vaccinate(ss.Vaccine(efficacy=0), on=d.outcome.to(d.D), name='vx'), solver='ode', total_pop=5000, unit='day', stop=60, **kw).run().results
    assert np.isclose(ode.vx.delivered.sum(), 0.1*ode.d.outcome[:-1].sum(), rtol=0.01)
    d, infected = make_seir()
    with pytest.raises(ss.CapabilityError):
        ss.Sim(d, interventions=ss.vaccinate(ss.Vaccine(efficacy=0), on=infected.enter), solver='ode', **kw).run()
    with pytest.raises(ValueError):
        d.outcome.to(d.E)
    return sim


if __name__ == '__main__':
    sc.options(interactive=False)
    test_events()
