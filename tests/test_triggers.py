"""
Test triggered moves (ss.on), the rest staying (ss.stay), lazy handles (ss.people, ss.state, ss.any), and distributions by group
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss

kw = dict(verbose=0)


@sc.timer()
def test_lazy():
    sc.heading('Testing lazy conditions on people, ss.state, and ss.any')
    def make(cond):
        S, I = ss.states('S I')
        d = ss.Disease(S, I, onset=ss.Transition(S, I, rate=ss.peryear(0.5), where=cond), name='d')
        return ss.Sim(d, n_agents=2000, start=2000, stop=2005, unit='year', **kw).run().diseases.d
    new = make(ss.People.female & (ss.People.age >= 15)) # Before there are any people
    assert new.I.count() > 100
    ppl = new.sim.people
    assert np.all(ppl.female[new.I.uids]) and np.all(ppl.age[new.I.uids] >= 15)

    # A state by path, for a module made before its siblings exist, and a union of several
    def pair(where):
        a = ss.Disease(sa=ss.progression('S -> I', rate=ss.peryear(0.3)), name='a')
        b = ss.Disease(sb=ss.progression('S -> I', rate=ss.peryear(1.0), where=where), name='b')
        return ss.Sim([a, b], n_agents=2000, start=2000, stop=2005, unit='year', **kw).run()
    sim = pair(ss.any(ss.state(f'{g}.I') for g in ['a']))
    a, b = sim.diseases.a, sim.diseases.b
    assert np.all(a.I[b.I.uids]) and b.I.count() > 100
    with pytest.raises(ValueError):
        ss.state('X')
    with pytest.raises(ValueError):
        ss.any([])
    return sim


if __name__ == '__main__':
    sc.options(interactive=False)
    test_lazy()
