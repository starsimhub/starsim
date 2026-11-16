"""
Test attributes (ss.attr): values by event
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss

kw = dict(verbose=0)


def test_attr_by_event():
    """ A dict by event records each event's own value, which a dwell time can read (Typhoid's prepatent period by dose) """
    class M(ss.Disease):
        S, R = ss.states('S R')
        E, I = ss.State(keep_time=True), ss.State(keep_time=True)
        cfu = ss.State(people=False, init=4.0)
        hi = ss.par(10.0)
        contact = S >> E @ ss.perday(0.05)*I
        environment = ss.Transition(S, E, rate=ss.perday(0.01), acquisition=True)
        dose = ss.attr({contact: hi, environment: cfu.n/2}) # By contact, a high dose; from the environment, what is there then
        onset = ss.Transition(E, I, dur=ss.lognormal(log_mean=np.log(dose), log_std=0.01, unit='day')) # The dwell is the dose, in days
        rec = I >> R @ ss.days(5)
    sim = ss.Sim(diseases=M(init_prev=0), n_agents=3000, dur=ss.days(40), dt=ss.days(1), rand_seed=1, **kw).run()
    m, u = sim.diseases.m, sim.people.auids
    dose = m.dose.raw[u]
    by_contact, by_env = sim.results.m.contact.sum(), sim.results.m.environment.sum()
    assert by_contact > 0 and by_env > 0 and np.sum(dose == 10) == by_contact and np.sum(dose == 2) == by_env # Each event's own value
    for d in [2, 10]:
        entered = u[(dose == d) & (m.I.raw[u] | m.R.raw[u])]
        assert len(entered) and np.allclose(m.ti_I.raw[entered] - m.ti_E.raw[entered], d, atol=1) # The dwell read its own dose
    with pytest.raises(ValueError):
        ss.attr({M.contact: 1}, when=M.contact) # Its keys are its events
    return sim


if __name__ == '__main__':
    T = sc.timer()
    test_attr_by_event()
    T.toc()
