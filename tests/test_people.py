"""
Test the People object
"""
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import starsim.library as ssl

sc.options.interactive = False # Assume not running interactively
small = 100


def test_people():
    """ Extra states, dead agents, agents that stand for fewer people, and arrays that take no memory until set """
    ppl = ss.People(small, extra_states=[ss.FloatArr('geolocation', default=lambda n: np.random.choice([1, 2, 3], n))])
    ppl.add_module(ssl.diseases.HIV()) # Possible outside a sim (not typical workflow)

    sim = ss.Sim(n_agents=small, diseases=dict(type='sir', init_prev=0.2, p_death=0.5), networks='random', verbose=0).run()
    dead = sim.people.dead # Dead agents are removed from other arrays, but people.dead covers all agents
    assert len(dead) == sim.people.n_uids > len(sim.people) and dead.sum() == sim.results.death.cum[-1] > 0
    assert not np.isin(dead.uids, sim.people.auids).any()

    S, I = ss.states('S I') # Agents that stand for fewer people (people.scale, as HPVsim's finer agents) count for less in the results, rates, and outcomes
    sis = ss.Disease(S, I, infection=S >> I @ ss.peryear(2)*I.frac, recovery=I >> S @ ss.years(1000), init=dict(I=0.5), name='sis')
    sim = ss.Sim(sis, n_agents=1000, start=2000, stop=2001, verbose=0).init()
    sim.people.scale[sim.diseases.sis.I.uids] = 0.01
    sim.run()
    weights = sim.people.scale.raw[sim.people.auids]
    assert np.isclose(sim.results.alive[-1], weights.sum()) and sim.results.sis.I[0] < 10 # Those infected at the start count for 0.01 each
    assert sim.results.sis.infection[-1] < 50 # I.frac is about 0.01, not 0.5: about 10 infections (unweighted, about 316)
    S, I = ss.states('S I')
    infection = S >> I @ ss.perday(5e-4)
    sim = ss.Sim(ss.Disease(S, I, infection, cases=ss.outcome(infection), name='sis'), n_agents=2000, stop=100, dt=1, unit='day', solver='abm', verbose=0).init()
    sim.people.scale[sim.people.auids] = 0.05
    res = sim.run().results.sis
    assert res.infection.sum() > 0 and np.isclose(res.cases.sum(), res.infection.sum()) # A step's events can be less than one, and aren't rounded away

    sim = ss.Sim(ss.SIR(), people=ss.People(small), networks='random', verbose=0).run() # Arrays nothing sets take no memory
    ppl, sir = sim.people, sim.diseases.sir
    assert 'auids' not in sim._before_init.pars.people.__dict__ # A People not yet in a sim has no arrays per agent
    assert ppl.slot.shared is ppl.uid and ppl.parent.shared and ppl.scale.shared and sir.rel_sus.shared
    assert ppl.auids.dtype == np.int64 and ppl.ti_dead.raw.dtype == np.float32
    ppl.scale[ppl.auids[:10]] = 2 # Setting one (as a Starsim 3 module does) gives it its own values
    assert ppl.scale.weighted and ppl.scale.shared is None and ppl.scale[ppl.auids[:11]].sum() == 21
    return ppl


def test_groups_are_not_agents():
    """ People mean people in a compartmental run: groups, not agents """
    S, I, R = ss.states(3)
    sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init_prev=0.01, name='sir')
    sim = ss.Sim(sir, total_pop=5000, stop=10, solver='ode', verbose=0).run()
    ppl = sim.people
    assert np.isclose(ppl.n, 5000) and len(ppl) == 1 and ppl.alive.values.all() # One group; arrays still work, on the groups
    for attr in ['age', 'auids']:
        with pytest.raises(ss.CapabilityError):
            getattr(ppl, attr)
    comp = ss.Sim(sir.stratify(age={0: 0.3, 15: 0.7}), total_pop=1000, unit='day', stop=60, verbose=0).run()
    assert comp.pars.n_agents is None and len(comp.people) == 2
    with pytest.raises(ss.CapabilityError): # count() is the number of agents
        comp.sir.I.count()
    return sim


def test_people_contains():
    """ `in` the people: a property by name, or an agent by UID """
    sim = ss.Sim(n_agents=small, verbose=0).init()
    ppl = sim.people
    assert 'age' in ppl and 'nothing' not in ppl
    assert 5 in ppl and small + 5 not in ppl
    with pytest.raises(TypeError):
        ppl[5] in ppl
    return


if __name__ == '__main__':
    T = sc.timer()
    test_people()
    test_groups_are_not_agents()
    test_people_contains()
    T.toc()
