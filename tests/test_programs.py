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


def make_sis(name='sis', beta=0.25):
    S, I = ss.states(2)
    return ss.Disease(infection=S >> I @ beta*I, recovery=I >> S @ ss.days(20), cure=I >> S, init_prev=0.01, name=name)


def test_queue():
    sc.heading('Testing queues: dropped, waiting up to a limit, and leaving when no longer eligible')
    def make(queue=None, **kwargs):
        sis = make_sis()
        clinic = ss.treat(ss.Treatment(moves=sis.cure), on=sis.infection, where=sis.I, capacity=ss.perday(20), queue=queue)
        return ss.Sim(sis, interventions=clinic, **kwargs)
    out = sc.objdict()
    for label,queue in dict(none=None, days=ss.days(5), true=True).items():
        sims = run_both(make, stop=150, queue=queue)
        for solver,sim in sims.items():
            r = sim.results.clinic
            n = sc.objdict({k:r[k].sum() for k in ['eligible', 'delivered', 'dropped'] + (['left'] if queue is not None else [])})
            print(f'{queue} {solver}: {n}, waiting at the end {r.waiting[-2] if queue is not None else 0:.0f}')
            assert np.all(r.delivered <= 20*1.0001) and r.delivered.max() > 19.9
            if queue is None:
                assert np.isclose(n.eligible, n.delivered + n.dropped)
            else:
                assert np.isclose(n.eligible, n.delivered + n.dropped + n.left + r.waiting[-2], rtol=1e-6) # Everyone who arrived is accounted for
                assert n.left > 0 and r.waiting.max() > 0
                assert r.wait.max() <= (5 if queue is not True else 150)
                if label == 'days':
                    assert np.allclose(r.wait[20:100], 5) # First come, first served: the oldest, who have waited the longest allowed
            out[f'{label}_{solver}'] = n
        check_close(sims, lambda sim: sim.results.clinic.dropped.sum() + 1, rtol=0.1)
    for solver in ['ode', 'abm']:
        assert out[f'true_{solver}'].dropped == 0 and out[f'days_{solver}'].dropped < 0.8*out[f'none_{solver}'].dropped
    return out


if __name__ == '__main__':
    sc.options(interactive=False)
    people = test_people()
    priority = test_priority()
    queue = test_queue()
