"""
Test programs: numbers are people, priority, queues, shared resources, stock targets, memory and doses, and steps at the same visit
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
    sc.heading('Testing priority: tiers, a score highest first, and a score per agent')
    def make(priority=None, **kwargs):
        sir = make_sir().stratify(risk=['low', 'high'], split=[0.8, 0.2]).stratify(age=[0, 15, 65])
        stock = ss.vaccinate(ss.Vaccine(efficacy=0.9), capacity=ss.perday(500), priority=priority(sir) if priority else None)
        return ss.Sim(sir, interventions=stock, **kwargs)
    sims = run_both(make, stop=40, priority=lambda sir: [sir.risk == 'high']) # All of the high-risk first (10,000 people, 20 days)
    for solver,sim in sims.items():
        by = sim.results.stock.delivered.by('risk').values
        assert np.allclose(by[:19, 1], 500) and np.allclose(by[:19, 0], 0), f'{solver}: {by[:21]}'
        assert by[21:39, 0].sum() > 0.95*500*18
    sims = run_both(make, stop=40, priority=lambda sir: ss.ppl.age) # Oldest first
    for solver,sim in sims.items():
        by = sim.results.stock.delivered.by('age').values
        n_old = sim.results.sir.n_S.by('age').values[0, 2] + sim.results.sir.n_I.by('age').values[0, 2]
        first = int(n_old//500) - 1
        assert np.allclose(by[:first, 2], 500) and by[:first, :2].sum() == 0, f'{solver}: {by[:first+2]}'
    check_close(sims, lambda sim: sim.results.stock.delivered.sum(), rtol=0.01)

    # A score per agent: under an explicit compartmental solver it is ignored with a note; with approx=False it is an error; left to choose, the model runs as agents
    uids = lambda sir: lambda sim, uids: -uids
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
            n = sc.objdict({k:r[k].sum() for k in ['eligible', 'delivered', 'lost_capacity'] + (['left'] if queue is not None else [])})
            print(f'{queue} {solver}: {n}, waiting at the end {r.waiting[-2] if queue is not None else 0:.0f}')
            assert np.all(r.delivered <= 20*1.0001) and r.delivered.max() > 19.9
            if queue is None:
                assert np.isclose(n.eligible, n.delivered + n.lost_capacity)
            else:
                assert np.isclose(n.eligible, n.delivered + n.lost_capacity + n.left + r.waiting[-2], rtol=1e-6) # Everyone who arrived is accounted for
                assert n.left > 0 and r.waiting.max() > 0
                assert r.waited.max() <= (5 if queue is not True else 150)
                if label == 'days':
                    assert np.allclose(r.waited[20:100], 5) and np.isclose(r.never_ran.sum(), n.lost_capacity) # First come, first served: the oldest, who have waited the longest allowed
            out[f'{label}_{solver}'] = n
        check_close(sims, lambda sim: sim.results.clinic.lost_capacity.sum() + 1, rtol=0.1)
    for solver in ['ode', 'abm']:
        assert out[f'true_{solver}'].lost_capacity == 0 and out[f'days_{solver}'].lost_capacity < 0.8*out[f'none_{solver}'].lost_capacity
    return out


def test_resource():
    sc.heading('Testing a resource shared by steps: by their class (priority), then pro rata')
    def make(ranks=(1, 2), **kwargs):
        sis = make_sis()
        clinic = ss.Resource(ss.perday(300))
        drug = ss.Treatment(moves=sis.cure)
        symptomatic = ss.treat(drug, where=sis.I, uses=clinic, priority=ranks[0])
        screening = ss.screen(ss.Diagnostic(sensitivity=0.9), coverage=0.1, interval=1, uses=clinic, priority=ranks[1])
        treatment = ss.treat(drug, who=screening.positive)
        return ss.Sim(sis, interventions=[symptomatic, screening, treatment], **kwargs)
    sims = run_both(make, stop=60)
    for solver,sim in sims.items():
        res = sim.results
        used = res.symptomatic.delivered + res.screening.delivered
        assert np.allclose(res.clinic.used, used) and np.allclose(res.clinic.available[:-1], 300)
        assert np.allclose(res.clinic.fraction_used[:-1], used[:-1]/300)
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
        prog = ss.treat(ss.Treatment(moves=tx.start), target=target, above=tx.dropout)
        return ss.Sim(tx, interventions=prog, **kwargs)
    sims = run_both(make, stop=150)
    for solver,sim in sims.items():
        r = sim.results.prog
        assert np.allclose(r.holding[1:149], r.target[1:149], atol=5.1), f'{solver}: {np.abs(r.holding - r.target)[1:149].max()}' # Within an agent (5 people)
        assert np.allclose(r.holding[10:149], sim.results.tx.n_T[11:150], rtol=0.015) # The number on treatment at the start of each step, less those who stop during it (about 0.5%, give or take an agent of 5 people)
        assert r.stopped[50:100].sum() > 0 and r.stopped[:50].sum() == 0 and r.shortfall.max() < 5.1
        assert any('counts tx.T' in note for note in sim.notes)
    check_close(sims, lambda sim: sim.results.prog.stopped.sum(), rtol=0.15) # Agents vary by about 6% between runs

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
        adults = ss.ppl.age >= 15
        vmmc = ss.vaccinate(ss.Vaccine(efficacy=0.6), target=ss.share(ss.timeseries({0: 0.2, 20: 0.6}, mode='linear'), among=adults), coverage=0.5)
        return ss.Sim(sir, interventions=vmmc, **kwargs)
    for solver,sim in run_both(shared, stop=30).items():
        r = sim.results.vmmc
        n_adults = sim.results.sir.n_S.by('age').values[0, 1:].sum() + sim.results.sir.n_I.by('age').values[0, 1:].sum()
        assert np.isclose(r.holding[25], 0.6*n_adults, rtol=0.02) and np.isclose(r.target[10], 0.4*n_adults, rtol=0.02)
        assert np.isclose(r.delivered.by('age').values[:, 0].sum(), 0) # Only adults
    return sims


def test_target_stop():
    sc.heading('Testing above= a target that falls: "stop" follows it down, lowest priority first, and a rate drifts down')
    n = ss.timeseries({0: 0, 20: 4000, 40: 1000}, mode='linear') # People: up, then down
    def make(above='stop', **kwargs):
        sir = make_sir().stratify(risk=['low', 'high'], split=[0.8, 0.2])
        prog = ss.vaccinate(ss.Vaccine(efficacy=0.9), target=n, above=above, priority=[sir.risk == 'high'], name='prog')
        return ss.Sim(sir, interventions=prog, **kwargs)
    for solver,sim in run_both(make, stop=50, pop=10_000).items():
        r = sim.results.prog
        assert np.allclose(r.holding[1:50], r.target[1:50], rtol=0.01, atol=5.1), f'{solver}: {np.abs(r.holding - r.target)[1:50].max()}'
        stopped = r.stopped.by('risk').values.sum(axis=0) # 3000 stopped: all 2000 of the low risk first, then 1000 of the high
        assert np.allclose(stopped, [2000, 1000], atol=5.1), f'{solver}: {stopped}'
    for solver,sim in run_both(make, stop=50, pop=10_000, above=ss.perday(0.05)).items(): # At a rate, toward the target
        r = sim.results.prog
        assert r.target[45] < r.holding[45] < r.holding[30] < r.holding[20] and r.stopped.by('risk').values[:, 1].sum() == 0, solver
    return sim


def test_target_limits():
    sc.heading('Testing a target within a capacity, and a target on a counter')
    def make(**kwargs):
        prog = ss.vaccinate(ss.Vaccine(efficacy=0.9), target=2000, capacity=ss.perday(100), coverage=0.5, name='prog')
        return ss.Sim(make_sir(), interventions=prog, **kwargs)
    for solver,sim in run_both(make, stop=30, pop=10_000).items(): # 100 a day toward 2000: only the gap not served is lost, not everyone chosen
        r = sim.results.prog
        assert np.allclose(r.holding[:19], 100*np.arange(1, 20), atol=5.1) and np.allclose(r.holding[20:29], 2000, atol=5.1), solver
        assert np.allclose(r.lost_capacity[:19], 2000 - r.holding[:19], atol=5.1) and r.lost_capacity[20:].sum() == 0, solver
    def counter(**kwargs): # A target on a counter counts those it has reached
        camp = ss.Delivery(increment='doses', coverage=0.5, target=3000, name='camp')
        sim = ss.Sim(make_sir(), interventions=camp, **kwargs)
        return sim.stratify(doses=[0, 1, 2])
    for solver,sim in run_both(counter, stop=10, pop=10_000).items():
        assert np.allclose(sim.results.camp.holding[1:9], 3000, atol=5.1), solver
    with pytest.raises(ValueError, match='goes down'):
        ss.Sim(make_sir(), interventions=ss.Delivery(increment=dict(doses=-1), target=10), verbose=0).stratify(doses=[0, 1]).init()
    return sim


def test_memory():
    sc.heading('Testing delivery memory: once per spell, .ever, .since, .held, and doses')

    # Routine delivery is once per spell of being eligible: each infection is offered treatment once; with ~drug.ever, once ever
    def spells(once=False, **kwargs):
        sis = make_sis(beta=0.15)
        drug = ss.Treatment(moves=sis.cure)
        clinic = ss.treat(drug, who=sis.I & ~drug.ever if once else sis.I, coverage=0.3)
        return ss.Sim(sis, interventions=clinic, **kwargs)
    each = run_both(spells, stop=200)
    once = run_both(spells, stop=200, once=True)
    for solver in ['ode', 'abm']:
        r, res = each[solver].results.clinic, each[solver].results.sis
        assert np.isclose(r.delivered[1:].sum(), 0.3*res.infection[:-1].sum(), rtol=0.1), f'{solver}: {r.delivered[1:].sum()} vs {0.3*res.infection[:-1].sum()}' # As agents, those whose dwell rounds to no steps recover in the step they are infected, before the clinic sees them (a few %)
        assert once[solver].results.clinic.delivered.sum() < 0.6*r.delivered.sum()
    check_close(each, lambda sim: sim.results.clinic.delivered.sum(), rtol=0.15) # As agents, a few % fewer are seen (above), and the epidemic is a few % smaller

    # Not within 30 days of the last time, at a rate per year; and the time since another delivery in a condition
    def rescreen(**kwargs):
        sis = make_sir()
        test = ss.Diagnostic(sensitivity=0.9)
        screening = ss.screen(test, who=test.since >= ss.days(30), coverage=ss.peryear(4))
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
            screened = sim.interventions.screening.last_event.raw
            assert np.nanmin(sim.ti - screened) >= 0
    check_close(sims, lambda sim: sim.results.screening.delivered.sum(), rtol=0.05)
    check_close(sims, lambda sim: sim.results.recent.delivered.sum(), rtol=0.05)
    assert 'since_screening' in [s.name for s in sims.ode.groups.strats] and 'since_screening' not in [s.name for s in sims.abm.groups.strats]

    # Coverage as a function that gives a typed rate is a rate at every step too, as the typed rate itself is
    def typed(func=False, **kwargs):
        rate = ss.peryear(4)
        screening = ss.screen(ss.Diagnostic(sensitivity=0.9), coverage=(lambda sim: rate) if func else rate, name='screening')
        return ss.Sim(make_sir(), interventions=screening, **kwargs)
    for solver in ['ode', 'abm']:
        direct, func = [run_both(typed, stop=60, func=f)[solver].results.screening.delivered for f in [False, True]]
        assert np.allclose(func, direct), f'{solver}: a function giving a typed rate should be the rate, not a routine delivery once per spell'
        assert func[30:].sum() > 0.5*func[:30].sum()

    # Two doses: the second at least 30 days after the first (each delivery to a holder gives the next dose); and .held
    def doses(**kwargs):
        sir = make_sir()
        vx = ss.Vaccine(efficacy=0.8, doses=[0.5, 1.0])
        first = ss.vaccinate(vx, coverage=0.6, at=[5])
        second = ss.vaccinate(vx, who=vx.ever & (vx.since >= ss.days(30)), coverage=0.5, interval=1)
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
    assert [s.labels for s in sims.ode.groups.strats if s.name == 'has_vx'][0] == [False, 'dose 1', 'dose 2']
    return sims


def test_same_visit():
    sc.heading('Testing steps at the same visit: a second and a third product for the people a delivery reached')
    def make(**kwargs):
        sis = make_sis()
        drug = ss.Treatment(moves={sis.cure: 0.9})
        visit = ss.treat(drug, where=sis.I, coverage=0.3, interval=ss.days(10))
        visit_then1 = ss.vaccinate(ss.Vaccine(efficacy=0.5), who=visit)
        visit_then2 = ss.screen(ss.Diagnostic(), who=visit, coverage=0.5)
        return ss.Sim(sis, interventions=[visit, visit_then1, visit_then2], **kwargs)
    sims = run_both(make, stop=60, pop=200_000, n_agents=200_000)
    for solver,sim in sims.items():
        res = sim.results
        assert np.isclose(res.visit_then1.delivered[0], res.visit.delivered[0]) # Everyone treated, none of whom hold it yet
        assert 0 < res.visit_then1.delivered.sum() <= res.visit.delivered.sum()
        assert np.isclose(res.visit_then2.delivered.sum(), 0.5*res.visit.delivered.sum(), rtol=0.05)
        assert np.isclose(res.visit_then2.positive.sum(), 0.1*res.visit_then2.delivered.sum(), rtol=0.15) # Tested at the same visit, after the drug (which cures 90%)
    check_close(sims, lambda sim: sim.results.visit_then1.delivered.sum(), rtol=0.1)
    return sims


# %% Acceptance tests: the worked examples of proposal 02, small

def test_hiv_art():
    sc.heading('Acceptance: HIV ART as a stock target, lowest CD4 first, with removal over the target (hiv/port.py art_deliver)')
    def make(prioritize=True, **kwargs):
        S, acute, cd4_hi, cd4_mid, cd4_lo, on_art, post_art = ss.states('S acute cd4_hi cd4_mid cd4_lo on_art post_art') # CD4 as states, so it runs both ways
        hiv = ss.Disease(
            infection  = S >> acute @ ss.permonth(0.08)*(5*acute + cd4_hi + cd4_mid + cd4_lo + post_art + 0.05*on_art),
            latency    = acute >> cd4_hi @ ss.months(3),
            decline    = cd4_hi >> cd4_mid @ ss.years(4),
            falling    = cd4_mid >> cd4_lo @ ss.years(3),
            aids_death = cd4_lo >> ss.null @ ss.years(2),
            start_art  = (acute + cd4_hi + cd4_mid + cd4_lo + post_art) >> on_art, # Undriven: only the delivery moves people along it
            dropout    = on_art >> post_art @ ss.years(5),
            relapse    = post_art >> cd4_lo @ ss.years(1),
            init=dict(cd4_hi=0.04, cd4_mid=0.03, cd4_lo=0.02), name='hiv',
        )
        n_art = ss.timeseries({2000: 0, 2005: 3000, 2010: 6000, 2012: 4000}, mode='linear') # People, from data
        priority = [hiv.cd4_lo, hiv.post_art, hiv.cd4_mid, hiv.cd4_hi] if prioritize else None # Lowest CD4 first
        art_prog = ss.treat(ss.Treatment(moves=hiv.start_art), target=n_art, priority=priority, above=hiv.dropout)
        return ss.Sim(hiv, interventions=art_prog, start=2000, stop=2014, dt=ss.months(1), unit='year', **kwargs) # The bare numbers (the years) are years
    kwa = dict(verbose=0)
    sims = sc.objdict(ode=make(solver='ode', total_pop=100_000, **kwa).run(), abm=make(solver='abm', total_pop=100_000, n_agents=20_000, **kwa).run())
    base = sc.objdict(ode=make(prioritize=False, solver='ode', total_pop=100_000, **kwa).run(), abm=make(prioritize=False, solver='abm', total_pop=100_000, n_agents=20_000, **kwa).run())
    for solver,sim in sims.items():
        r, res, res0 = sim.results.art_prog, sim.results.hiv, base[solver].results.hiv
        assert np.allclose(r.holding[1:-1], r.target[1:-1], atol=10.1), f'{solver}: {np.abs(r.holding - r.target)[1:-1].max():.1f}' # Within two agents (as agents, a dwell on ART drawn shorter than a step ends at once)
        assert r.stopped[132:144].sum() > 0 and r.stopped[:120].sum() == 0 # Over the target in 2011-2012 only
        k = 30 # The lowest CD4 is treated first: fewer left in it than at random, and more in the highest
        assert res0.n_cd4_lo[k] - res.n_cd4_lo[k] > 0.5*r.target[k] and res.n_cd4_hi[k] > res0.n_cd4_hi[k]
        assert any('counts hiv.on_art' in note for note in sim.notes)
    check_close(sims, lambda sim: sim.results.hiv.aids_death.sum(), rtol=0.15)
    return sims


def test_hpv_algorithm():
    sc.heading('Acceptance: HPV screen, then a treatment visit with a capacity of 10 a week and a queue (hpvsim_interventions/port.py algo_a, one genotype)')
    def make(**kwargs):
        S, I, CIN, CA = ss.states('S I CIN CA')
        hpv = ss.Disease(
            infection = S >> I @ 0.6*(I + CIN),
            clearance = I >> S @ ss.years(1.5),
            dysplasia = I >> CIN @ ss.years(5),
            regress   = CIN >> I @ ss.years(4),
            invasion  = CIN >> CA @ ss.years(15),
            excised   = CIN >> I, # Undriven: clears the lesion, not the infection
            init=dict(I=0.1, CIN=0.03), name='hpv',
        ).stratify(age=[0, 30, 50])
        women = (ss.ppl.age >= 30) & (ss.ppl.age < 50)
        hpv_dna = ss.Diagnostic(sensitivity=dict(I=0.9, CIN=0.95), specificity=0.97)
        primary = ss.screen(hpv_dna, who=women & (hpv_dna.since >= ss.years(5)), coverage=ss.probperyear(0.3), start=2020)
        tx_visit = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.9, detects=~hpv.CA), who=primary.positive, coverage=0.8, capacity=ss.perweek(10), queue=ss.months(6))
        ablate = ss.treat(ss.Treatment(moves={hpv.excised: 0.85}, name='ablation'), who=tx_visit.positive)
        excise = ss.treat(ss.Treatment(moves={hpv.excised: 0.95}, name='excision'), who=tx_visit.negative)
        return ss.Sim(hpv, interventions=ss.cascade(primary, tx_visit, ablate, excise), start=2015, stop=2030, dt=ss.months(3), use_aging=False, **kwargs)
    kwa = dict(verbose=0)
    sims = sc.objdict(ode=make(solver='ode', total_pop=100_000, **kwa).run(), abm=make(solver='abm', total_pop=100_000, n_agents=20_000, **kwa).run())
    for solver,sim in sims.items():
        res = sim.results
        q = res.tx_visit
        n = sc.objdict({k:q[k].sum() for k in ['eligible', 'delivered', 'never_ran', 'left']})
        print(f'{solver}: {n}, waiting at the end {q.waiting[-2]:.0f}')
        cap = 10*365.25/4/7 # 10 a week, about 13 weeks a quarter
        full = q.waiting[25:-1] > 1 # Steps with people queueing, when the capacity is all used
        assert np.all(q.delivered <= cap + 5.1) and full.sum() > 5 and np.isclose(q.delivered[25:-1][full].mean(), cap, rtol=0.02) # Within an agent, as agents
        assert q.waiting.max() > 0 and n.never_ran > 0 and q.waited.max() <= 6 + 1e-6, q.waited.max() # In the sim's unit, months
        assert np.isclose(0.8*n.eligible, n.delivered + n.never_ran + n.left + q.waiting[-2], rtol=1e-6 if solver == 'ode' else 0.05) # Of those who come (coverage 0.8; as agents, a few % low, the known coverage bias)
        assert np.isclose(res.ablate.delivered.sum() + res.excise.delivered.sum(), n.delivered, rtol=0.01)
        assert res.ablate.unneeded.sum() > 0 and res.ablate.cured.sum() > 0
        women = res.primary.eligible[20] # Women 30-49, none screened yet at the start of 2020
        assert res.primary.delivered[:-1].sum() < 2*women # Nobody is screened twice in 5 years, so at most twice in 2020-2030
    check_close(sims, lambda sim: sim.results.primary.delivered.sum(), rtol=0.05)
    check_close(sims, lambda sim: sim.results.tx_visit.never_ran.sum(), rtol=0.2)
    assert any('since' in s.name for s in sims.ode.groups.strats) and any('waiting' in s.name for s in sims.ode.groups.strats)
    return sims


def make_sti(name='ng', beta=0.015, init_prev=0.1):
    """ A bacterial STI as in STIsim (stisim_bacterial/port.py), with treatment as an undriven transition """
    S, E, A, P, Y, C = ss.states('S E A P Y C')
    return ss.Disease(
        infection = S >> E @ beta*(A + P + Y + C),
        latency   = ss.progression('E -> P | A', p=0.4, dur=ss.days(5)),
        onset     = P >> Y @ ss.days(10),
        seek      = Y >> C @ ss.days(20),
        clear_a   = A >> S @ ss.days(150),
        clear_y   = (Y + C) >> S @ ss.days(200),
        cure      = (E + A + P + Y + C) >> S,
        init_prev=init_prev, name=name,
    )


def test_clinic():
    sc.heading('Acceptance: one clinic for syndromic treatment and screening, symptomatic first (stisim_bacterial)')
    def make(**kwargs):
        ng, ct = make_sti('ng', beta=0.015), make_sti('ct', beta=0.012)
        clinic = ss.Resource(ss.perweek(250))
        discharge = ss.Treatment(moves={ng.cure: 0.95, ct.cure: 0.95})
        syndromic = ss.treat(discharge, where=ng.C | ct.C, uses=clinic, queue=ss.days(28), priority=1)
        screening = ss.screen(ss.Diagnostic(sensitivity=0.9), disease='ng', coverage=ss.probperyear(0.5), uses=clinic, priority=2)
        treatment = ss.treat(discharge, who=screening.positive)
        return ss.Sim([ng, ct], interventions=[syndromic, screening, treatment], dt=7, **kwargs)
    sims = run_both(make, stop=364, pop=20_000, n_agents=20_000)
    for solver,sim in sims.items():
        res = sim.results
        used = res.syndromic.delivered + res.screening.delivered
        assert np.allclose(res.clinic.used, used) and np.all(used <= 250.01)
        full = res.syndromic.delivered > 249.99
        assert full.any() and np.all(res.screening.delivered[full] < 1e-9) # Screening only gets what symptomatic care leaves
        assert res.screening.delivered.sum() > 0
        r = res.syndromic
        for dis in ['ng', 'ct']:
            assert np.isclose(sum(r[k].select(disease=dis).sum() for k in ['cured', 'failed', 'unneeded']), r.delivered.sum())
        assert r.unneeded.select(disease='ct').sum() > 0 and r.waiting.max() > 0
    check_close(sims, lambda sim: sim.results.syndromic.delivered.sum(), rtol=0.15)
    check_close(sims, lambda sim: sim.results.screening.delivered.sum(), rtol=0.25)
    return sims


def test_rescreen():
    sc.heading('Acceptance: rescreening every 5 years at most, at 30% a year')
    def make(**kwargs):
        sir = make_sir()
        test = ss.Diagnostic()
        primary = ss.screen(test, who=test.since >= ss.years(5), coverage=ss.probperyear(0.3), count_events=True)
        return ss.Sim(sir, interventions=primary, start=2020, stop=2040, dt=ss.months(3), **kwargs)
    kwa = dict(verbose=0)
    sims = sc.objdict(ode=make(solver='ode', total_pop=10_000, **kwa).run(), abm=make(solver='abm', total_pop=10_000, n_agents=10_000, **kwa).run())
    df = sims.abm.interventions.primary.events_df()
    tested = df[df.event == 'primary'].sort_values(['uid', 'time'])
    assert tested.groupby('uid').time.diff().dropna().min() >= np.timedelta64(5*365, 'D') # As agents: nobody screened twice within 5 years
    for solver,sim in sims.items():
        d, e = sim.results.primary.delivered, sim.results.primary.eligible
        assert np.isclose(d[0], 10_000*(1 - 0.7**0.25), rtol=0.05)
        assert np.isclose(e[19], 10_000 - d[:19].sum()) and np.isclose(e[20], 10_000 - d[1:20].sum()) # Those screened first are due again after exactly 5 years
    check_close(sims, lambda sim: sim.results.primary.delivered.sum(), rtol=0.03)
    assert 'since_test' in [s.name for s in sims.ode.groups.strats]
    return sims


def test_two_doses():
    sc.heading('Acceptance: a two-dose vaccine, the second at least 6 months after the first (hpvsim_interventions/port.py)')
    def make(**kwargs):
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=ss.peryear(1.5)), recovery=ss.progression('I -> S', dur=ss.years(1)), init_prev=0.05, name='sis').stratify(age=[0, 9, 15])
        vx = ss.Vaccine(efficacy=0.9, doses=[0.6, 1.0])
        girls = (ss.ppl.age >= 9) & (ss.ppl.age < 15)
        first = ss.vaccinate(vx, who=girls & ~vx.ever, coverage=ss.probperyear(0.5), start=2020)
        second = ss.vaccinate(vx, who=vx.ever & (vx.since >= ss.months(6)), coverage=0.9) # The next dose, for those who had the first
        return ss.Sim(sir, interventions=[first, second], start=2019, stop=2026, dt=ss.months(1), use_aging=False, **kwargs)
    kwa = dict(verbose=0)
    sims = sc.objdict(ode=make(solver='ode', total_pop=50_000, **kwa).run(), abm=make(solver='abm', total_pop=50_000, n_agents=25_000, **kwa).run())
    for solver,sim in sims.items():
        res = sim.results
        d1, d2 = res.first.delivered, res.second.delivered
        start = int(np.flatnonzero(d1)[0])
        assert d2[:start + 6].sum() == 0 and d2[start + 6] > 0 # No second dose within 6 months of the first
        assert np.isclose(d2[start + 6:].sum(), 0.9*d1[:-7].sum(), rtol=0.05)
        assert np.isclose(res.first.delivered.by('age').values[:, [0, 2]].sum(), 0)
    check_close(sims, lambda sim: sim.results.second.delivered.sum(), rtol=0.05)
    check_close(sims, lambda sim: sim.results.sis.n_I[-2], rtol=0.1)
    return sims


if __name__ == '__main__':
    sc.options(interactive=False)
    people = test_people()
    priority = test_priority()
    queue = test_queue()
    resource = test_resource()
    target = test_target()
    tstop = test_target_stop()
    limits = test_target_limits()
    memory = test_memory()
    visit = test_same_visit()
    hiv = test_hiv_art()
    hpv = test_hpv_algorithm()
    clinic = test_clinic()
    rescreen = test_rescreen()
    doses = test_two_doses()
