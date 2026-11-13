"""
Test programs: numbers are people, priority, queues, memory and doses, and steps at the same visit
"""

# %% Imports and settings
import pytest
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(verbose=0, unit='day')


def make_sir():
    S, I, R = ss.states('S I R')
    return ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')


def run_both(make, stop=100, pop=50_000, n_agents=10_000, **kwargs):
    """ Run the sim that `make` returns as an ODE and as agents (each agent standing for several people) """
    return sc.obj(ode=make(solver='ode', total_pop=pop, stop=stop, **kw, **kwargs).run(),
                      abm=make(solver='abm', total_pop=pop, n_agents=n_agents, stop=stop, **kw, **kwargs).run())


def check_close(sims, func, rtol=0.1):
    vals = sc.obj({solver:func(sim) for solver,sim in sims.items()})
    assert np.allclose(vals[:], vals.ode, rtol=rtol), f'Expected the solvers to agree, not {vals}'
    return vals


def check_agents(make, stop=100, pop=50_000, **kwargs):
    """ Check that a model is agents only: an explicit ODE refuses it before the run, and solver='auto' runs it as agents """
    with pytest.raises(ss.CapabilityError):
        make(solver='ode', total_pop=pop, stop=stop, **kw, **kwargs).init()
    assert isinstance(make(total_pop=pop, stop=stop, **kw, **kwargs).init().solver, ss.solvers.abm)
    return


def run_abm(make, stop=100, pop=50_000, n_agents=10_000, **kwargs):
    return sc.obj(abm=make(solver='abm', total_pop=pop, n_agents=n_agents, stop=stop, **kw, **kwargs).run())


# %% Define the tests

def test_people():
    sc.heading('Testing numbers are people: capacity and results read during a run')
    seen = sc.obj()
    def make(**kwargs):
        sir = make_sir()
        def look(sim): # Results read during the run are people, with 5 people to an agent
            if sim.ti == 30:
                seen[sim.solver.name] = sc.obj(I=sim.results.sir.I[sim.ti], delivered=sim.results.stock.delivered[sim.ti - 1]) # The state at the start of the step, and the deliveries of the last
            return True
        stock = ss.vaccinate(ss.Vaccine(efficacy=0.9), who=look, when=ss.schedule(start=10), capacity=ss.perday(500))
        return ss.Sim(sir, interventions=stock, **kwargs)
    sims = run_both(make, stop=60)
    for solver,sim in sims.items():
        res = sim.results
        assert np.allclose(res.stock.delivered[10:59], 500, rtol=0.02), f'{solver}: {res.stock.delivered[10:20]}' # From the step that starts at 10
        assert seen[solver].I == res.sir.I[30] and seen[solver].delivered == res.stock.delivered[29] # Not scaled again at the end
    assert sims.abm.pop_scale == 5
    check_close(sims, lambda sim: seen[sim.solver.name].I, rtol=0.15)
    check_close(sims, lambda sim: sim.results.sir.infection.cum[-1], rtol=0.1)
    return sims


def test_priority():
    sc.heading('Testing priority: tiers, a score highest first, and a score per agent')
    def make(priority=None, **kwargs):
        sir = make_sir().stratify(risk={'low': 0.8, 'high': 0.2}).stratify(age=[0, 15, 65])
        stock = ss.vaccinate(ss.Vaccine(efficacy=0.9), capacity=ss.capacity(ss.perday(500), priority=priority(sir) if priority else None))
        return ss.Sim(sir, interventions=stock, **kwargs)
    sims = run_both(make, stop=40, priority=lambda sir: [sir.risk == 'high']) # All of the high-risk first (10,000 people, 20 days)
    for solver,sim in sims.items():
        by = sim.results.stock.delivered.by('risk').values
        assert np.allclose(by[0:20, 1], 500) and np.allclose(by[0:20, 0], 0), f'{solver}: {by[:22]}'
        assert by[21:39, 0].sum() > 0.95*500*18
    sims = run_both(make, stop=40, priority=lambda sir: ss.person.age) # Oldest first
    for solver,sim in sims.items():
        by = sim.results.stock.delivered.by('age').values
        n_old = sim.results.sir.S.by('age').values[0, 2] + sim.results.sir.I.by('age').values[0, 2]
        first = int(n_old//500) - 1
        assert np.allclose(by[0:first, 2], 500) and by[0:first, :2].sum() == 0, f'{solver}: {by[:first+3]}'
    check_close(sims, lambda sim: sim.results.stock.delivered.sum(), rtol=0.01)

    # A score per agent: under an explicit compartmental solver it is ignored with a note; with approximations='error' it is an error; left to choose, the model runs as agents
    uids = lambda sir: lambda sim, uids: -uids
    keyed = make(priority=uids, solver='ode', total_pop=50_000, stop=10, **kw).run()
    assert any('priority ignored' in note for note in keyed.notes)
    with pytest.raises(ss.CapabilityError):
        make(priority=uids, solver='ode', approximations='error', total_pop=50_000, stop=10, **kw).run()
    assert isinstance(make(priority=uids, total_pop=50_000, stop=10, **kw).init().solver, ss.solvers.abm)
    lowest = make(priority=uids, solver='abm', total_pop=50_000, n_agents=10_000, stop=3, **kw).run()
    got = lowest.people.has_vaccine.raw[:10_000].astype(bool)
    assert got[:400].all() and not got[400:].any() # 100 agents a day for the 4 steps (days 0 to 3), lowest UID first
    return sims


def make_sis(name='sis', beta=0.25):
    S, I = ss.states(2)
    return ss.Disease(infection=S >> I @ beta*I, recovery=I >> S @ ss.days(20), cure=I >> S @ 0, init_prev=0.01, name=name)


def test_queue():
    sc.heading('Testing queues: dropped, waiting up to a limit (agents only), and leaving when no longer eligible')
    def make(queue=None, **kwargs):
        sis = make_sis()
        clinic = ss.treat(ss.Treatment(cure={sis.cure: 1.0}), who=sis.I, when=sis.infection, capacity=ss.capacity(ss.perday(20), queue=queue))
        return ss.Sim(sis, interventions=clinic, **kwargs)
    out = sc.obj()
    for label,queue in dict(none=None, days=ss.days(5), true=True).items():
        if queue is None:
            sims = run_both(make, stop=150, queue=queue)
        else: # Who waits, and since when, is each person's own
            check_agents(make, stop=150, queue=queue)
            sims = run_abm(make, stop=150, queue=queue)
        for solver,sim in sims.items():
            r = sim.results.clinic
            n = sc.obj({k:r[k].sum() for k in ['eligible', 'delivered', 'lost_capacity'] + (['left'] if queue is not None else [])})
            print(f'{queue} {solver}: {n}, waiting at the end {r.waiting[-1] if queue is not None else 0:.0f}')
            assert np.all(r.delivered <= 20*1.0001) and r.delivered.max() > 19.9
            if queue is None:
                assert np.isclose(n.eligible, n.delivered + n.lost_capacity)
            else:
                assert np.isclose(n.eligible, n.delivered + n.lost_capacity + n.left + r.waiting[-1], rtol=1e-6) # Everyone who arrived is accounted for
                assert n.left > 0 and r.waiting.max() > 0
                assert r.waited.max() <= (5 if queue is not True else 150)
                if label == 'days':
                    assert np.allclose(r.waited[20:100], 5) and np.isclose(r.never_ran.sum(), n.lost_capacity) # First come, first served: the oldest, who have waited the longest allowed
            out[f'{label}_{solver}'] = n
        if queue is None:
            check_close(sims, lambda sim: sim.results.clinic.lost_capacity.sum() + 1, rtol=0.1)
    assert out.true_abm.lost_capacity == 0 and out.days_abm.lost_capacity < 0.8*out.none_abm.lost_capacity
    return out


def test_memory():
    sc.heading('Testing delivery memory: once per spell, .ever, .since, .held, and doses')

    # Routine delivery is once per spell of being eligible: each infection is offered treatment once; with ~drug.ever, once ever
    def spells(once=False, **kwargs):
        sis = make_sis(beta=0.15)
        drug = ss.Treatment(cure={sis.cure: 1.0})
        clinic = ss.treat(drug, who=sis.I & ~drug.ever if once else sis.I, coverage=0.3)
        return ss.Sim(sis, interventions=clinic, **kwargs)
    each = run_both(spells, stop=200)
    once = run_both(spells, stop=200, once=True)
    for solver in ['ode', 'abm']:
        r, res = each[solver].results.clinic, each[solver].results.sis
        assert np.isclose(r.delivered[1:].sum(), 0.3*res.infection[:-1].sum(), rtol=0.1), f'{solver}: {r.delivered[1:].sum()} vs {0.3*res.infection[:-1].sum()}' # As agents, those whose dwell rounds to no steps recover in the step they are infected, before the clinic sees them (a few %)
        assert once[solver].results.clinic.delivered.sum() < 0.6*r.delivered.sum()
    check_close(each, lambda sim: sim.results.clinic.delivered.sum(), rtol=0.15) # As agents, a few % fewer are seen (above), and the epidemic is a few % smaller

    # Not within 30 days of the last time, at a rate per year; and the time since another delivery in a condition (.since is agents only)
    def rescreen(**kwargs):
        sis = make_sir()
        test = ss.Diagnostic(sensitivity=0.9)
        screening = ss.screen(test, who=test.since >= ss.days(30), coverage=ss.peryear(4))
        recent = ss.vaccinate(ss.Vaccine(efficacy=0.5), who=screening.since < ss.days(5), coverage=0.1)
        return ss.Sim(sis, interventions=[screening, recent], **kwargs)
    check_agents(rescreen, stop=120)
    sim = run_abm(rescreen, stop=120).abm
    r = sim.results.screening
    p = 1 - np.exp(-4/365) # Each day, of those not screened in the last 30 days
    assert np.isclose(r.coverage[0:30].mean(), p, rtol=0.07) and np.isclose(r.eligible[1], 50_000*(1 - p), rtol=0.01) # Over 30 steps, since as agents one step varies by about 10%; as agents it is also about 4% low (as on v4-rs: a known issue)
    assert np.isclose(r.eligible[60], 50_000 - r.delivered[31:60].sum(), rtol=0.01) # Those screened in the 30 days before are not eligible
    assert sim.results.recent.delivered[0] > 0 and sim.results.recent.eligible[50] < 6*r.delivered[45:50].sum()
    screened = sim.interventions.screening.last_event.raw
    assert np.nanmin(sim.ti - screened) >= 0

    # Coverage as a function that gives a typed rate is a rate at every step too, as the typed rate itself is
    def typed(func=False, **kwargs):
        rate = ss.peryear(4)
        screening = ss.screen(ss.Diagnostic(sensitivity=0.9), coverage=(lambda sim: rate) if func else rate, name='screening')
        return ss.Sim(make_sir(), interventions=screening, **kwargs)
    runs = [run_both(typed, stop=60, func=f) for f in [False, True]]
    for solver in ['ode', 'abm']:
        direct, func = [r[solver].results.screening.delivered for r in runs]
        assert np.allclose(func, direct), f'{solver}: a function giving a typed rate should be the rate, not a routine delivery once per spell'
        assert func[30:].sum() > 0.5*func[:30].sum()

    # Two doses: the second at least 30 days after the first (each delivery to a holder gives the next dose; agents only, by .since), or in a round of its own (every solver); and .held
    def doses(since=True, **kwargs):
        sir = make_sir()
        vx = ss.Vaccine(efficacy=0.8, doses=[0.5, 1.0])
        first = ss.vaccinate(vx, when=[5], coverage=0.6)
        second = ss.vaccinate(vx, who=vx.ever & (vx.since >= ss.days(30)), when=ss.schedule(every=1), coverage=0.5) if since else ss.vaccinate(vx, who=vx.ever, when=[35], coverage=0.5, name='second')
        held = ss.vaccinate(ss.Vaccine(efficacy=0.1, name='other'), who=vx.held, when=[60])
        return ss.Sim(sir, interventions=[first, second, held], **kwargs)
    check_agents(doses, stop=80)
    for since,sims in [(True, run_abm(doses, stop=80)), (False, run_both(doses, stop=80, since=False))]:
        for solver,sim in sims.items():
            res = sim.results
            assert np.isclose(res.first.delivered[5], 30_000, rtol=0.02)
            d2 = res.second.delivered
            assert d2[:35].sum() == 0 and np.isclose(d2[35], 15_000, rtol=0.03) # Half of those who had the first
            assert not since or np.isclose(d2.sum(), res.first.delivered.sum(), rtol=0.01) # Everyone who had the first, in the end
            assert np.isclose(res.held.delivered[60], 30_000, rtol=0.02) # Everyone who had a dose
    check_close(sims, lambda sim: sim.results.sir.infection.cum[-1], rtol=0.2) # Agents are about 10% below the ODE at a step of a day (as in test_transfer), and one run varies too
    assert [s.labels for s in sims.ode.groups.strats if s.name == 'has_vx'][0] == [False, 'dose 1', 'dose 2']
    return sims


def test_same_visit():
    sc.heading('Testing steps at the same visit: a second and a third product for the people a delivery reached')
    def make(**kwargs):
        sis = make_sis()
        drug = ss.Treatment(cure={sis.cure: 0.9})
        visit = ss.treat(drug, who=sis.I, when=ss.schedule(every=ss.days(10)), coverage=0.3)
        visit_then1 = ss.vaccinate(ss.Vaccine(efficacy=0.5), when=visit)
        visit_then2 = ss.screen(ss.Diagnostic(), when=visit, coverage=0.5)
        return ss.Sim(sis, interventions=[visit, visit_then1, visit_then2], **kwargs)
    sims = run_both(make, stop=60, pop=200_000, n_agents=200_000)
    for solver,sim in sims.items():
        res = sim.results
        assert np.isclose(res.visit_then1.delivered[1], res.visit.delivered[1]) # Everyone treated, none of whom hold it yet
        assert 0 < res.visit_then1.delivered.sum() <= res.visit.delivered.sum()
        assert np.isclose(res.visit_then2.delivered.sum(), 0.5*res.visit.delivered.sum(), rtol=0.05)
        assert np.isclose(res.visit_then2.positive.sum(), 0.1*res.visit_then2.delivered.sum(), rtol=0.15) # Tested at the same visit, after the drug (which cures 90%)
    check_close(sims, lambda sim: sim.results.visit_then1.delivered.sum(), rtol=0.1)
    return sims


# %% Worked examples, small

def test_hiv_art():
    sc.heading('Testing HIV ART within a capacity, lowest CD4 first')
    def make(prioritize=True, **kwargs):
        S, acute, cd4_hi, cd4_mid, cd4_lo, on_art, post_art = ss.states('S acute cd4_hi cd4_mid cd4_lo on_art post_art') # CD4 as states, so it runs both ways
        hiv = ss.Disease(
            infection  = S >> acute @ ss.permonth(0.08)*(5*acute + cd4_hi + cd4_mid + cd4_lo + post_art + 0.05*on_art),
            latency    = acute >> cd4_hi @ ss.months(3),
            decline    = cd4_hi >> cd4_mid @ ss.years(4),
            falling    = cd4_mid >> cd4_lo @ ss.years(3),
            aids_death = cd4_lo >> ss.dead @ ss.years(2),
            start_art  = (acute + cd4_hi + cd4_mid + cd4_lo + post_art) >> on_art @ 0, # Only the delivery moves people along it
            dropout    = on_art >> post_art @ ss.years(5),
            relapse    = post_art >> cd4_lo @ ss.years(1),
            init=dict(cd4_hi=0.04, cd4_mid=0.03, cd4_lo=0.02), name='hiv',
        )
        priority = [hiv.cd4_lo, hiv.post_art, hiv.cd4_mid, hiv.cd4_hi] if prioritize else None # Lowest CD4 first
        art_prog = ss.treat(ss.Treatment(moves=hiv.start_art), capacity=ss.capacity(ss.peryear(1200), priority=priority))
        return ss.Sim(hiv, interventions=art_prog, start=2000, stop=2014, dt=ss.months(1), unit='year', **kwargs) # The bare numbers (the years) are years
    both = lambda prioritize: sc.obj(ode=make(prioritize, solver='ode', total_pop=100_000, verbose=0).run(), abm=make(prioritize, solver='abm', total_pop=100_000, n_agents=20_000, verbose=0).run())
    sims, base = both(True), both(False)
    for solver,sim in sims.items():
        r, res, res0 = sim.results.art_prog, sim.results.hiv, base[solver].results.hiv
        assert np.allclose(r.delivered[1:], 100, atol=5.1), f'{solver}: {r.delivered[1:].min():.1f}' # 100 a month, within an agent
        k = 30 # The lowest CD4 is treated first: fewer left in it than at random, and more in the highest
        assert res.cd4_lo[k] < 0.8*res0.cd4_lo[k] and res.cd4_hi[k] > res0.cd4_hi[k], f'{solver}: {res.cd4_lo[k]:.0f} vs {res0.cd4_lo[k]:.0f}'
    check_close(sims, lambda sim: sim.results.hiv.aids_death.sum(), rtol=0.15)
    return sims


def make_sti(name='ng', beta=0.015, init_prev=0.1):
    """ A bacterial STI as in STIsim, with treatment as an undriven transition """
    S, E, A, P, Y, C = ss.states('S E A P Y C')
    return ss.Disease(
        infection = S >> E @ beta*(A + P + Y + C),
        latency   = ss.Transition(E, {P: 0.4, A: None}, dur=ss.days(5)),
        onset     = P >> Y @ ss.days(10),
        seek      = Y >> C @ ss.days(20),
        clear_a   = A >> S @ ss.days(150),
        clear_y   = (Y + C) >> S @ ss.days(200),
        cure      = (E + A + P + Y + C) >> S @ 0,
        init_prev=init_prev, name=name,
    )


def test_clinic():
    sc.heading('Testing one clinic for syndromic treatment and screening, symptomatic first, as agents')
    def make(**kwargs):
        ng, ct = make_sti('ng', beta=0.015), make_sti('ct', beta=0.012)
        discharge = ss.Treatment(cure={ng.cure: 0.95, ct.cure: 0.95})
        syndromic = ss.treat(discharge, who=ng.C | ct.C, capacity=ss.capacity(ss.perweek(200), queue=ss.days(28)))
        left = lambda sim: 200 - sim.interventions.syndromic.results.delivered[sim.ti] # What symptomatic care leaves of the clinic's week (it runs first)
        screening = ss.screen(ss.Diagnostic(sensitivity=0.9), disease='ng', coverage=ss.probperyear(0.5), capacity=left)
        treatment = ss.treat(discharge, when=screening.positive)
        return ss.Sim([ng, ct], interventions=[syndromic, screening, treatment], dt=7, **kwargs)
    check_agents(make, stop=364, pop=20_000) # The queue
    sim = run_abm(make, stop=364, pop=20_000, n_agents=20_000).abm
    res = sim.results
    used = res.syndromic.delivered + res.screening.delivered
    assert np.all(used <= 200.01)
    full = res.syndromic.delivered > 199.99
    assert full.any() and np.all(res.screening.delivered[full] < 1e-9) # Screening only gets what symptomatic care leaves
    assert res.screening.delivered.sum() > 0
    r = res.syndromic
    for dis in ['ng', 'ct']:
        assert np.isclose(sum(r[k].select(disease=dis).sum() for k in ['cured', 'failed', 'unneeded']), r.delivered.sum())
    assert r.unneeded.select(disease='ct').sum() > 0 and r.waiting.max() > 0
    return sim


if __name__ == '__main__':
    sc.options(interactive=False)
    people = test_people()
    priority = test_priority()
    queue = test_queue()
    memory = test_memory()
    visit = test_same_visit()
    hiv = test_hiv_art()
    clinic = test_clinic()
