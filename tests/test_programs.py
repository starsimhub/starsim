"""
Test programs (proposal 02): numbers are people, priority, queues, shared capacity, stock targets, delivery memory, and then=
"""

# %% Imports and settings
import pytest
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(verbose=0, unit='day')


def make_sir():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')


def run_both(make, stop=100, pop=50_000, n_agents=10_000, **kwargs):
    """ Run the sim that `make` returns as an ODE and as agents (each agent standing for several people) """
    return sc.objdict(ode=make(solver='ode', total_pop=pop, stop=stop, **kw, **kwargs).run(),
                      abm=make(solver='abm', total_pop=pop, n_agents=n_agents, stop=stop, **kw, **kwargs).run())


def check_close(sims, func, rtol=0.1):
    vals = sc.objdict({solver:func(sim) for solver,sim in sims.items()})
    assert np.allclose(vals[:], vals.ode, rtol=rtol), f'Expected the solvers to agree, not {vals}'
    return vals


# %% Define the tests

def test_people():
    sc.heading('Testing numbers are people: capacity and results read during a run')
    seen = sc.objdict()
    def make(**kwargs):
        sir = make_sir()
        def look(sim): # Results read during the run are people, with 5 people to an agent
            if sim.ti == 30:
                seen[sim.solver] = sc.objdict(n_I=sim.results.sir.n_I[sim.ti - 1], delivered=sim.results.stock.delivered[sim.ti - 1])
            return True
        stock = ss.vaccinate(ss.Vaccine(efficacy=0.9), capacity=ss.perday(500), start=10, when=look)
        return ss.Sim(sir, interventions=stock, **kwargs)
    sims = run_both(make, stop=60)
    for solver,sim in sims.items():
        res = sim.results
        assert np.allclose(res.stock.delivered[10:59], 500, rtol=0.02), f'{solver}: {res.stock.delivered[10:20]}'
        assert seen[solver].n_I == res.sir.n_I[29] and seen[solver].delivered == res.stock.delivered[29] # Not scaled again at the end
    assert sims.abm.pars.pop_scale == 5
    check_close(sims, lambda sim: seen[sim.solver].n_I, rtol=0.15)
    check_close(sims, lambda sim: sim.results.sir.infection.cum[-1], rtol=0.1)
    return sims


def test_priority():
    sc.heading('Testing priority: tiers, a key highest first, and a per-agent key')
    ppl = ss.People(1)
    def make(priority=None, **kwargs):
        sir = make_sir().stratify(risk=['low', 'high'], split=[0.8, 0.2]).stratify(age=[0, 15, 65])
        stock = ss.vaccinate(ss.Vaccine(efficacy=0.9), capacity=ss.perday(500), priority=priority(sir) if priority else None)
        return ss.Sim(sir, interventions=stock, **kwargs)
    sims = run_both(make, stop=40, priority=lambda sir: [sir.risk == 'high']) # All of the high-risk first (10,000 people, 20 days)
    for solver,sim in sims.items():
        by = sim.results.stock.delivered.by('risk').values
        assert np.allclose(by[:19, 1], 500) and np.allclose(by[:19, 0], 0), f'{solver}: {by[:21]}'
        assert by[21:39, 0].sum() > 0.95*500*18
    sims = run_both(make, stop=40, priority=lambda sir: ss.desc(ppl.age)) # Oldest first
    for solver,sim in sims.items():
        by = sim.results.stock.delivered.by('age').values
        n_old = sim.results.sir.n_S.by('age').values[0, 2] + sim.results.sir.n_I.by('age').values[0, 2]
        first = int(n_old//500) - 1
        assert np.allclose(by[:first, 2], 500) and by[:first, :2].sum() == 0, f'{solver}: {by[:first+2]}'
    check_close(sims, lambda sim: sim.results.stock.delivered.sum(), rtol=0.01)

    # A key per agent: under an explicit compartmental solver it is ignored with a note; with approx=False it is an error; left to choose, the model runs as agents
    uids = lambda sir: lambda sim, uids: uids
    keyed = make(priority=uids, solver='ode', total_pop=50_000, stop=10, **kw).run()
    assert any('priority ignored' in note for note in keyed.notes)
    with pytest.raises(ss.CapabilityError):
        make(priority=uids, solver='ode', approx=False, total_pop=50_000, stop=10, **kw).run()
    assert make(priority=uids, total_pop=50_000, stop=10, **kw).init().solver == 'abm'
    lowest = make(priority=uids, solver='abm', total_pop=50_000, n_agents=10_000, stop=3, **kw).run()
    got = lowest.people.has_vaccine.raw[:10_000].astype(bool)
    assert got[:300].all() and not got[300:].any() # 100 agents a day, lowest UID first
    return sims


if __name__ == '__main__':
    sc.options(interactive=False)
    people = test_people()
    priority = test_priority()
