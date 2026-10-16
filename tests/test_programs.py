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


def test_capacity():
    sc.heading('Testing a capacity shared by deliveries: by their priority, then pro rata')
    def make(ranks=(1, 2), **kwargs):
        sis = make_sis()
        clinic = ss.Capacity(ss.perday(300))
        drug = ss.Treatment(moves=sis.cure)
        symptomatic = ss.treat(drug, where=sis.I, capacity=clinic, priority=ranks[0])
        screening = ss.screen(ss.Test(sensitivity=0.9), coverage=0.1, interval=1, capacity=clinic, priority=ranks[1], on_positive=ss.treat(drug))
        return ss.Sim(sis, interventions=[symptomatic, screening], **kwargs)
    sims = run_both(make, stop=60)
    for solver,sim in sims.items():
        res = sim.results
        used = res.symptomatic.delivered + res.screening.delivered
        assert np.allclose(res.clinic.used, used) and np.allclose(res.clinic.available[:-1], 300)
        assert np.allclose(res.clinic.unused[:-1], 300 - used[:-1])
        full = res.symptomatic.delivered[:-1] > 299.99 # Symptomatic first: screening gets what is left
        assert full[0] and np.all(res.screening.delivered[:-1][full] < 1e-9)
        assert np.allclose(used[:-1], 300) and not full.all() # The rest of the clinic's time is spent on screening
        assert res.screening.delivered.sum() > 1000
    check_close(sims, lambda sim: sim.results.screening.delivered.sum(), rtol=0.15)

    # Equal priorities: shared pro rata, by what each would serve
    for solver,sim in run_both(make, stop=10, ranks=(0, 0)).items():
        res = sim.results
        n_I = res.sis.n_I[0]
        share = res.symptomatic.delivered[0]/(res.symptomatic.delivered[0] + res.screening.delivered[0])
        expected = n_I/(n_I + 0.1*50_000)
        assert np.isclose(share, expected, rtol=0.1), f'{solver}: {share:.3f} vs {expected:.3f}'
    return sims


def make_tx(name='tx'):
    """ Infection, and treatment as a state: started only by a delivery, and stopped at a rate """
    S, I, T = ss.states('S I T')
    return ss.Disease(infection=S >> I @ 0.002*(I + T), start=I >> T, dropout=T >> I @ ss.days(200), init=dict(I=0.4), name=name)


def test_target():
    sc.heading('Testing stock targets: the gap delivered each step, removal over it, by stratum, and as a share')
    n_on = ss.timeseries({0: 0, 50: 8000, 100: 4000}, mode='linear') # People
    def make(target=n_on, **kwargs):
        tx = make_tx()
        prog = ss.treat(ss.Treatment(moves=tx.start), target=target, remove=tx.dropout)
        return ss.Sim(tx, interventions=prog, **kwargs)
    sims = run_both(make, stop=150)
    for solver,sim in sims.items():
        r = sim.results.prog
        assert np.allclose(r.n_end[1:149], r.target[1:149], atol=5.1), f'{solver}: {np.abs(r.n_end - r.target)[1:149].max()}' # Within an agent (5 people)
        assert np.allclose(r.n_end[10:149], sim.results.tx.n_T[11:150], rtol=0.015) # The number on treatment at the start of each step, less those who stop during it (about 0.5%, give or take an agent of 5 people)
        assert r.removed[50:100].sum() > 0 and r.removed[:50].sum() == 0 and r.shortfall.max() < 5.1
        assert any('counts tx.T' in note for note in sim.notes)
    check_close(sims, lambda sim: sim.results.prog.removed.sum(), rtol=0.15) # Agents vary by about 6% between runs

    # By stratum, and a share of some people
    def by_risk(**kwargs):
        tx = make_tx().stratify(risk=['low', 'high'], split=[0.5, 0.5])
        prog = ss.treat(ss.Treatment(moves=tx.start), target=ss.by(risk=[1000, 3000]))
        return ss.Sim(tx, interventions=prog, **kwargs)
    for solver,sim in run_both(by_risk, stop=30).items():
        n_T = sim.results.tx.n_T.by('risk').values
        assert np.allclose(n_T[2:29], [1000, 3000], rtol=0.02), f'{solver}: {n_T[2:5]}' # Less those who stopped in the step (about 0.5%), give or take an agent of 5 people
    def shared(**kwargs):
        sir = make_sir().stratify(age=[0, 15, 65])
        adults = ss.People.age >= 15
        vmmc = ss.vaccinate(ss.Vaccine(efficacy=0.6), target=ss.share(ss.timeseries({0: 0.2, 20: 0.6}, mode='linear'), among=adults), coverage=0.5)
        return ss.Sim(sir, interventions=vmmc, **kwargs)
    for solver,sim in run_both(shared, stop=30).items():
        r = sim.results.vmmc
        n_adults = sim.results.sir.n_S.by('age').values[0, 1:].sum() + sim.results.sir.n_I.by('age').values[0, 1:].sum()
        assert np.isclose(r.n_end[25], 0.6*n_adults, rtol=0.02) and np.isclose(r.target[10], 0.4*n_adults, rtol=0.02)
        assert np.isclose(r.delivered.by('age').values[:, 0].sum(), 0) # Only adults
    return sims


def test_memory():
    sc.heading('Testing delivery memory: once per spell, repeat=, .since, .held, and doses')

    # Routine delivery is once per spell of being eligible: each infection is offered treatment once; repeat=False is once ever
    def spells(repeat=None, **kwargs):
        sis = make_sis(beta=0.15)
        clinic = ss.treat(ss.Treatment(moves=sis.cure), where=sis.I, coverage=0.3, repeat=repeat)
        return ss.Sim(sis, interventions=clinic, **kwargs)
    each = run_both(spells, stop=200)
    once = run_both(spells, stop=200, repeat=False)
    for solver in ['ode', 'abm']:
        r, res = each[solver].results.clinic, each[solver].results.sis
        assert np.isclose(r.delivered[1:].sum(), 0.3*res.infection[:-1].sum(), rtol=0.1), f'{solver}: {r.delivered[1:].sum()} vs {0.3*res.infection[:-1].sum()}' # As agents, those whose dwell rounds to no steps recover in the step they are infected, before the clinic sees them (a few %)
        assert once[solver].results.clinic.delivered.sum() < 0.6*r.delivered.sum()
    check_close(each, lambda sim: sim.results.clinic.delivered.sum(), rtol=0.15) # As agents, a few % fewer are seen (above), and the epidemic is a few % smaller

    # Not within 30 days of the last time, at a rate per year; and the time since another delivery in a condition
    def rescreen(**kwargs):
        sis = make_sir()
        test = ss.Test(sensitivity=0.9)
        screening = ss.screen(test, coverage=ss.peryear(4), repeat=ss.days(30))
        recent = ss.vaccinate(ss.Vaccine(efficacy=0.5), where=screening.since < ss.days(5), coverage=0.1)
        return ss.Sim(sis, interventions=[screening, recent], **kwargs)
    sims = run_both(rescreen, stop=120)
    for solver,sim in sims.items():
        r = sim.results.screening
        p = 1 - np.exp(-4/365) # Each day, of those not screened in the last 30 days
        assert np.isclose(r.coverage[1:31].mean(), p, rtol=0.07) and np.isclose(r.eligible[1], 50_000*(1 - p), rtol=0.01) # Over 30 steps, since as agents one step varies by about 10%; as agents it is also about 4% low (as on v4-rs: a known issue)
        assert np.isclose(r.eligible[60], 50_000 - r.delivered[31:60].sum(), rtol=0.01) # Those screened in the 30 days before are not eligible
        assert sim.results.recent.delivered[1] > 0 and sim.results.recent.eligible[50] < 6*r.delivered[45:50].sum()
        if solver == 'abm':
            screened = sim.interventions.screening.ti_last.raw
            assert np.nanmin(sim.ti - screened) >= 0
    check_close(sims, lambda sim: sim.results.screening.delivered.sum(), rtol=0.05)
    check_close(sims, lambda sim: sim.results.recent.delivered.sum(), rtol=0.05)
    assert 'since_screening' in [s.name for s in sims.ode.strats] and 'since_screening' not in [s.name for s in sims.abm.strats]

    # Two doses: the second at least 30 days after the first; and .held
    def doses(**kwargs):
        sir = make_sir()
        vx = ss.Vaccine(efficacy=0.8, doses=[0.5, 1.0])
        first = ss.vaccinate(vx, dose=1, coverage=0.6, at=[5])
        second = ss.vaccinate(vx, dose=2, where=vx.since >= ss.days(30), coverage=0.5, interval=1)
        held = ss.vaccinate(ss.Vaccine(efficacy=0.1, name='other'), where=vx.held, at=[60])
        return ss.Sim(sir, interventions=[first, second, held], **kwargs)
    sims = run_both(doses, stop=80)
    for solver,sim in sims.items():
        res = sim.results
        assert np.isclose(res.first.delivered[5], 30_000, rtol=0.02)
        d2 = res.second.delivered
        assert d2[:35].sum() == 0 and np.isclose(d2[35], 15_000, rtol=0.03) and np.isclose(d2.sum(), res.first.delivered.sum(), rtol=0.01) # Everyone who had the first, in the end
        assert np.isclose(res.held.delivered[60], 30_000, rtol=0.02) # Everyone who had a dose
    check_close(sims, lambda sim: sim.results.sir.infection.cum[-1], rtol=0.2) # Agents are about 10% below the ODE at a step of a day (as in test_transfer), and one run varies too
    assert [s.labels for s in sims.ode.strats if s.name == 'has_vx'][0] == [False, 'dose 1', 'dose 2']
    with pytest.raises(ValueError):
        ss.vaccinate(ss.Vaccine(efficacy=0.8), dose=2)
    return sims


def test_then():
    sc.heading('Testing then=: a second product at the same visit, and a list of them')
    def make(**kwargs):
        sis = make_sis()
        drug = ss.Treatment(moves={sis.cure: 0.9})
        visit = ss.treat(drug, where=sis.I, coverage=0.3, interval=ss.days(10), then=[ss.vaccinate(ss.Vaccine(efficacy=0.5)), ss.screen(ss.Test(), coverage=0.5)])
        return ss.Sim(sis, interventions=visit, **kwargs)
    sims = run_both(make, stop=60, pop=200_000, n_agents=200_000)
    for solver,sim in sims.items():
        res = sim.results
        assert np.isclose(res.visit_then1.delivered[0], res.visit.delivered[0]) # Everyone treated, none of whom hold it yet
        assert 0 < res.visit_then1.delivered.sum() <= res.visit.delivered.sum()
        assert np.isclose(res.visit_then2.delivered.sum(), 0.5*res.visit.delivered.sum(), rtol=0.05)
        assert np.isclose(res.visit_then2.diagnosed.sum(), 0.1*res.visit_then2.delivered.sum(), rtol=0.15) # Tested at the same visit, after the drug (which cures 90%)
    check_close(sims, lambda sim: sim.results.visit_then1.delivered.sum(), rtol=0.1)

    # A function, as agents
    seen = []
    sim = ss.Sim(make_sis(), interventions=ss.treat(ss.Treatment(transition='cure'), coverage=0.2, at=[5], then=lambda sim, uids: seen.append(len(uids))), stop=10, n_agents=5000, **kw).run()
    assert sim.solver == 'abm' and seen == [sim.results.treat.delivered[5]]
    return sims


if __name__ == '__main__':
    sc.options(interactive=False)
    people = test_people()
    priority = test_priority()
    queue = test_queue()
    capacity = test_capacity()
    target = test_target()
    memory = test_memory()
    then = test_then()
