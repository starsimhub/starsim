"""
Test interventions on declared models: modifiers, conditions, products, and their delivery
"""

# %% Imports and settings
import pytest
import numpy as np
import networkx as nx
import sciris as sc
import starsim as ss

kw = dict(verbose=0, unit='day', total_pop=50_000)
solvers = ['ode', 'binomial', 'abm']


def make_sir(**kwargs):
    sir = ss.Disease(
        infection = ss.transmission('S -> I', beta=0.3),
        recovery  = ss.progression('I -> R', dur=10),
        init_prev = 0.01,
        name = 'sir',
        **kwargs,
    )
    return sir


def make_sis():
    sis = ss.Disease(
        infection = ss.transmission('S -> I', beta=0.3),
        recovery  = ss.progression('I -> S', dur=10),
        init_prev = 0.01,
        name = 'sis',
    )
    return sis


def run_each(make, stop=150, **kwargs):
    """ Run the sim that `make` returns with each solver """
    return sc.objdict({solver:make(solver=solver, stop=stop, **kw, **kwargs).run() for solver in solvers})


def check_close(sims, func, rtol=0.1):
    """ Check that a number from each sim matches the ODE's """
    vals = sc.objdict({solver:func(sim) for solver,sim in sims.items()})
    assert np.allclose(vals[:], vals.ode, rtol=rtol), f'Expected the solvers to agree, not {vals}'
    return vals


# %% Define the tests

def test_change():
    sc.heading('Testing ss.change()')
    def make(**kwargs):
        sir = make_sir()
        lockdown = ss.change(sir.pars.beta, at=[20, 60], by=[0.3, 1.0])
        return ss.Sim(sir, interventions=lockdown, **kwargs)
    sims = run_each(make, stop=120)
    base = ss.Sim(make_sir(), solver='ode', stop=120, **kw).run()
    res, res0 = sims.ode.results.sir, base.results.sir
    assert np.allclose(res.n_S[:20], res0.n_S[:20]) # The same until it starts, at step 20
    assert res.infection[20:60].sum() < 0.5*res0.infection[20:60].sum()
    assert res.infection[60:].sum() > res0.infection[60:].sum() # The epidemic resumes when the multiplier goes back to 1
    check_close(sims, lambda sim: sim.results.sir.cum_infection[-1])

    # Values instead of multipliers, and dates
    sir = make_sir()
    to = ss.change(sir.pars.beta, at=['2020-01-21', '2020-03-01'], to=[0.09, 0.3])
    dated = ss.Sim(sir, interventions=to, start='2020-01-01', stop='2020-04-30', total_pop=50_000, verbose=0).run()
    assert np.allclose(dated.results.sir.n_S, res.n_S)
    with pytest.raises(ValueError):
        ss.change(sir.pars.beta, at=[20, 60], by=[0.3])
    return sims


def test_modify_where():
    sc.heading('Testing modifiers with conditions, and targets other than parameters')
    def make(**kwargs):
        sir = make_sir().stratify(risk=['lower', 'higher'], split=[0.8, 0.2])
        risky = ss.modify(sir.rel_sus, multiply=3.0, where=sir.risk == 'higher')
        faster = ss.modify(sir.recovery, multiply=2.0, start=20)
        quieter = ss.modify(sir.rel_trans, multiply=lambda sim: 0.5 if sim.ti > 40 else 1.0)
        return ss.Sim(sir, connectors=risky, interventions=[faster, quieter], **kwargs)
    sims = run_each(make)
    for sim in sims.values():
        by_risk = sim.results.sir.infection.by('risk').values.sum(axis=0)
        assert by_risk[1]/0.2 > 1.1*by_risk[0]/0.8, 'Expected a higher attack rate in the higher-risk group'
    check_close(sims, lambda sim: sim.results.sir.cum_infection[-1])
    text = sims.ode.what_modifies('sir.rel_sus', output=True)
    assert 'risky' in text and 'multiply=3.0' in text and 'higher' in text
    assert 'faster' in sims.abm.what_modifies(sims.abm.diseases.sir.recovery, output=True)
    assert 'none' in sims.ode.what_modifies('sir.pars.beta', output=True)

    # A modifier of a parameter can depend on the group, but in a compartmental run not on the state
    sir = make_sir().stratify(risk=['lower', 'higher'])
    group = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.0, where=sir.risk == 'lower'), solver='ode', stop=50, **kw).run()
    assert np.isclose(group.results.sir.infection.by('risk').values[:, 0].sum(), 0)
    with pytest.raises(ss.CodedError, match='E804'):
        ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.5, where=sir.I), solver='ode', stop=50, **kw).run()

    # Two modifiers that set the same thing are noted
    sir = make_sir()
    sets = [ss.modify(sir.pars.beta, set=0.2, name='a'), ss.modify(sir.pars.beta, set=0.1, name='b')]
    both = ss.Sim(sir, interventions=sets, stop=20, **kw).run()
    assert any('both set' in note for note in both.notes)
    return sims


def test_conditions():
    sc.heading('Testing conditions')
    ppl = ss.People(100)
    sir = make_sir().stratify(age=[0, 15, 65])
    cond = (ppl.age < 15) & sir.S
    assert isinstance(cond, ss.Cond)
    assert str(cond) == '((people.age < 15) & sir.S)'
    with pytest.raises(TypeError):
        bool(cond)

    # The same condition selects agents or groups
    def make(eligible=cond, **kwargs):
        vaccine = ss.Vaccine(efficacy=0.9)
        kids = ss.vaccinate(vaccine, eligible=eligible, years=[5])
        return ss.Sim(sir, interventions=kids, **kwargs)
    sims = run_each(make, stop=20)
    for sim in sims.values():
        n_S = sim.results.sir.n_S.by('age').values[5, 0]
        assert np.isclose(sim.results.kids.delivered[5], n_S, rtol=0.02) # Every susceptible child, and nobody else
        assert np.allclose(sim.results.kids.delivered.by('age').values[5], [sim.results.kids.delivered[5], 0, 0]) # By stratum: all in the youngest group
    check_close(sims, lambda sim: sim.results.kids.delivered.sum(), rtol=0.05)

    # A function of the sim does the same
    func = make(eligible=lambda sim: sim.people.age < 15, solver='ode', stop=20, **kw).run()
    assert func.results.kids.delivered[5] > sims.ode.results.kids.delivered[5] # Including the children who are not susceptible

    # An age that is not the edge of a group cannot be used on groups, so the model runs as agents unless told otherwise
    auto = make(eligible=ppl.age < 10, stop=20, **kw).init()
    assert auto.solver == 'abm' and 'falls inside a bin' in auto.solver_reason
    with pytest.raises(ss.CodedError, match='E804'):
        make(eligible=ppl.age < 10, solver='ode', stop=20, **kw).run()
    return sims


def test_vaccinate():
    sc.heading('Testing vaccination: campaigns, routine delivery, waning, and capacity')
    base = ss.Sim(make_sir(), solver='ode', stop=150, **kw).run()

    # A campaign
    def campaign(**kwargs):
        sir = make_sir()
        vaccine = ss.Vaccine(efficacy=0.9, cost=2.0)
        rounds = ss.vaccinate(vaccine, coverage=0.6, years=[5])
        return ss.Sim(sir, interventions=rounds, **kwargs)
    sims = run_each(campaign)
    for sim in sims.values():
        res = sim.results.rounds
        assert np.isclose(res.delivered.sum(), 30_000, rtol=0.02) and np.isclose(res.coverage[5], 0.6, atol=0.02)
        assert np.isclose(res.cost.sum(), 2*res.delivered.sum())
        assert sim.results.sir.cum_infection[-1] < 0.5*base.results.sir.cum_infection[-1]
        assert [strat.name for strat in sim.strats] == ['has_vaccine'] # Holding the vaccine is a property of people
    assert sims.ode.hash().model == base.hash().model # The scenario is not part of the model
    assert 'has_vaccine' in sims.ode.describe(output=True)

    # Routine delivery: each person is offered it once, and half accept
    def routine(**kwargs):
        sir = make_sir()
        vaccine = ss.Vaccine(efficacy=0.9)
        offer = ss.vaccinate(vaccine, coverage=0.5, start=10)
        return ss.Sim(sir, interventions=offer, **kwargs)
    sims = run_each(routine)
    for sim in sims.values():
        res = sim.results.offer
        assert np.isclose(res.delivered[10], 25_000, rtol=0.03) and res.delivered[:10].sum() == 0 and res.delivered[11:].sum() == 0

    # Protection that is lost, and given again in the next round; with efficacy against two things
    def waning(**kwargs):
        sis = make_sis()
        vaccine = ss.Vaccine(efficacy=dict(acquisition=0.8, transmission=0.5), waning=ss.days(30))
        rounds = ss.vaccinate(vaccine, coverage=0.8, interval=ss.days(50), start=20)
        return ss.Sim(sis, interventions=rounds, **kwargs)
    sims = run_each(waning, stop=200)
    for sim in sims.values():
        delivered = sim.results.rounds.delivered
        assert list(np.flatnonzero(delivered)) == [20, 70, 120, 170]
        expected = 0.8*(10_000 + 40_000*(1 - np.exp(-50/30))) # Those missed in the last round, and those who have lost their protection since
        assert np.isclose(delivered[70], expected, rtol=0.05)
    check_close(sims, lambda sim: sim.results.sis.n_I[100])

    # A number available per day, while stocks last; and coverage by age
    def capacity(**kwargs):
        sir = make_sir().stratify(age=[0, 15, 65])
        vaccine = ss.Vaccine(efficacy=0.9)
        stock = ss.vaccinate(vaccine, capacity=ss.perday(500), coverage=ss.by(age=[1.0, 1.0, 0.0]), start=10, stop=40)
        return ss.Sim(sir, interventions=stock, **kwargs)
    sims = run_each(capacity, stop=60)
    for sim in sims.values():
        delivered = sim.results.stock.delivered
        assert np.allclose(delivered[10:40], 500, rtol=0.1) and delivered[:10].sum() == 0 and delivered[40:].sum() == 0

    # Errors
    with pytest.raises(ss.CodedError, match='E505'):
        ss.Sim(make_sir(), interventions=ss.vaccinate(ss.Vaccine(efficacy=dict(aquisition=0.5))), **kw).init()
    with pytest.raises(TypeError):
        ss.vaccinate(ss.Test())
    # Coverage and capacity that vary in time
    def varying(**kwargs):
        vaccine = ss.Vaccine(efficacy=0.9)
        supply = ss.vaccinate(vaccine, coverage=ss.timeseries({0: 0.0, 10: 0.05, 20: 0.0}), interval=1, capacity=ss.timeseries({0: 1e9, 15: 100}))
        return ss.Sim(make_sir(), interventions=supply, **kwargs)
    sims = run_each(varying, stop=30)
    for sim in sims.values():
        delivered = sim.results.supply.delivered
        assert delivered[:10].sum() == 0 and delivered[20:].sum() == 0 # Nobody is reached outside the ten days
        assert delivered[12] > 1500 and delivered[17] == 100 # About 5% of those left each day, until supplies are limited
    return sims


def test_transfer():
    sc.heading('Testing vaccination as a transfer along a stratification')
    def make(**kwargs):
        sir = make_sir().stratify(vaccinated=[False, True], split=[1, 0], adjust={'infection': [1.0, 0.15]})
        rollout = ss.transfer(vaccinated=True, coverage=0.7, years=[5, 30], eligible=~sir.vaccinated)
        return ss.Sim(sir, interventions=rollout, **kwargs)
    sims = run_each(make)
    for sim in sims.values():
        delivered = sim.results.rollout.delivered
        assert np.allclose(delivered[[5, 30]], [35_000, 10_500], rtol=0.05) # 70% of everyone, then 70% of those left
    check_close(sims, lambda sim: sim.results.sir.cum_infection[-1], rtol=0.2) # What is left after the campaign depends on how far the epidemic had got, which varies by about 10% between agent-based runs
    return sims


def test_treat():
    sc.heading('Testing treatment: curing, and changing rates')
    base = ss.Sim(make_sis(), solver='ode', stop=150, **kw).run()

    # A treatment that cures
    def cure(**kwargs):
        sis = make_sis()
        drug = ss.Treatment(efficacy=0.9)
        rounds = ss.treat(drug, coverage=0.3, interval=5, start=30)
        return ss.Sim(sis, interventions=rounds, **kwargs)
    sims = run_each(cure)
    for solver,sim in sims.items():
        res = sim.results
        assert res.sis.n_I[-1] < 0.9*base.results.sis.n_I[-1]
        assert np.isclose(res.rounds.delivered[30], 0.3*res.sis.n_I[30], rtol=0.07) # By default, the people who are infected at the time
    check_close(sims, lambda sim: sim.results.rounds.delivered.sum(), rtol=0.12) # As agents, about 9% fewer are infected (a difference of order dt)

    # A treatment that changes rates, with protection that fades by a function of time (which needs agents)
    def make_sird():
        return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), death=ss.death('I ->', rate=0.02), init_prev=0.01, name='sird')
    def rates(waning=None, **kwargs):
        sird = make_sird()
        art = ss.Treatment(efficacy=dict(transmission=0.5, mortality=0.9), waning=waning)
        program = ss.treat(art, eligible=sird.I, coverage=0.8)
        return ss.Sim(sird, interventions=program, **kwargs)
    untreated = ss.Sim(make_sird(), solver='ode', stop=150, **kw).run()
    sims = run_each(rates)
    for sim in sims.values():
        assert sim.results.sird.death.sum() < 0.6*untreated.results.sird.death.sum()
    check_close(sims, lambda sim: sim.results.program.delivered.sum()/sim.results.sird.cum_infection[-1]) # Most of the people infected, whenever that was
    fades = rates(waning=lambda t: np.exp(-t/5), stop=150, **kw).run()
    assert fades.solver == 'abm'
    assert fades.results.sird.death.sum() > 1.2*sims.abm.results.sird.death.sum()

    with pytest.raises(ss.CodedError, match='E505'):
        ss.Sim(make_sird(), interventions=ss.treat(ss.Treatment(efficacy=0.5, transition='cure')), **kw).init()
    return sims


def test_screen():
    sc.heading('Testing screening, follow-ups, and event-triggered delivery')

    # Screen in rounds, and treat most of those who test positive
    def rounds(**kwargs):
        sis = make_sis()
        test = ss.Test(sensitivity=0.9, specificity=0.95)
        drug = ss.Treatment(efficacy=0.95)
        screening = ss.screen(test, coverage=0.4, interval=7, start=30, on_positive=ss.treat(drug, coverage=0.8))
        return ss.Sim(sis, interventions=screening, **kwargs)
    sims = run_each(rounds)
    for solver,sim in sims.items():
        res = sim.results
        rtol = 0.12 if solver == 'abm' else 0.05 # As agents, the exits due in a step happen at its start, before the screening (here a tenth of the infected, with a dwell time of 10 days)
        ti = 37
        n_I = res.sis.n_I[ti] # At the time
        expected = 0.4*(0.9*n_I + 0.05*(50_000 - n_I)) # True positives and false positives
        assert np.isclose(res.screening.delivered[ti], 20_000, rtol=0.03)
        assert np.isclose(res.screening.diagnosed[ti], expected, rtol=rtol)
        assert np.isclose(res.screening_positive.delivered[ti], 0.8*res.screening.diagnosed[ti], rtol=0.05) # The follow-up is a module of its own
    check_close(sims, lambda sim: sim.results.screening.diagnosed.sum())

    # Test half of new infections, and cure those found
    def events(**kwargs):
        sir = make_sir()
        test = ss.Test(sensitivity=0.8)
        drug = ss.Treatment(efficacy=1.0)
        symptomatic = ss.screen(test, on=sir.infection, coverage=0.5, on_positive=ss.treat(drug))
        return ss.Sim(sir, interventions=symptomatic, **kwargs)
    sims = run_each(events)
    for sim in sims.values():
        res = sim.results
        assert np.isclose(res.symptomatic.delivered.sum(), 0.5*res.sir.infection[:-1].sum(), rtol=0.05)
        assert np.isclose(res.symptomatic.diagnosed.sum(), 0.8*res.symptomatic.delivered.sum(), rtol=0.08) # As agents, a few have already recovered by the time they are tested
    check_close(sims, lambda sim: sim.results.sir.cum_infection[-1])

    # A test with a delay needs agents; the follow-ups happen when the result arrives, and can be functions
    negatives = []
    def delay(**kwargs):
        sis = make_sis()
        test = ss.Test(sensitivity=0.9, delay=ss.days(3))
        drug = ss.Treatment(efficacy=1.0)
        slow = ss.screen(test, coverage=0.5, years=[40], on_positive=ss.treat(drug), on_negative=lambda sim, uids: negatives.append(len(uids)))
        return ss.Sim(sis, interventions=slow, **kwargs)
    sim = delay(stop=60, **kw).run()
    res = sim.results
    assert sim.solver == 'abm'
    assert list(np.flatnonzero(res.slow.delivered)) == [40] and list(np.flatnonzero(res.slow.diagnosed)) == [43]
    assert np.isclose(res.slow_positive.delivered[43], res.slow.diagnosed[43])
    assert negatives[0] > 0
    with pytest.raises(ss.CodedError, match='E804'):
        delay(solver='ode', stop=60, **kw).run()
    return sims


def test_deliver():
    sc.heading('Testing delivery by a function, triggers, and scenarios')

    # A function that says who gets it: UIDs for agents, a number per group otherwise
    def give(sim):
        if sim.ti != 5:
            return ss.uids() if sim.solver == 'abm' else np.zeros(sim.n_groups)
        elif sim.solver == 'abm':
            return sim.people.auids[:10_000]
        return sim.diseases.sir.S.values*0.5
    def by_function(**kwargs):
        sir = make_sir()
        vaccine = ss.Vaccine(efficacy=0.9)
        chosen = ss.vaccinate(vaccine, deliver=give, capacity=8000)
        return ss.Sim(sir, interventions=chosen, **kwargs)
    for solver in ['ode', 'abm']:
        sim = by_function(solver=solver, stop=20, **kw).run()
        assert np.isclose(sim.results.chosen.delivered.sum(), 8000) # Capped by the capacity
    with pytest.raises(ValueError):
        ss.vaccinate(ss.Vaccine(efficacy=0.9), deliver=give, coverage=0.5)

    # A trigger: vaccinate while there are many infections
    def triggered(**kwargs):
        sir = make_sir()
        vaccine = ss.Vaccine(efficacy=0.9)
        many = lambda sim: sim.results.sir.n_I[sim.ti - 1] > 2000
        response = ss.vaccinate(vaccine, coverage=0.2, interval=1, when=many)
        return ss.Sim(sir, interventions=response, **kwargs)
    sims = run_each(triggered)
    for sim in sims.values():
        res = sim.results
        given = np.flatnonzero(res.response.delivered)
        assert len(given) and np.all(res.sir.n_I[given - 1] > 2000)

    # Scenarios are sims, and can be compared run by run
    def scenario(vaccinate=True, **kwargs):
        sir = make_sir()
        campaign = ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.6, years=[5])
        return ss.Sim(sir, interventions=campaign if vaccinate else None, solver='binomial', stop=100, **kw)
    msim = ss.parallel(baseline=scenario(False), vaccine=scenario(True), n_runs=3)
    diff = msim.diff('baseline').vaccine.sir.cum_infection
    assert diff.values.shape == (3, 101)
    assert np.all(diff.values[:, -1] < 0) # Fewer infections in every run
    return msim


def test_coinfection():
    sc.heading('Testing two diseases that modify each other')
    def make(coinfect=True):
        hiv = ss.Disease(infection=ss.transmission('S -> I', beta=dict(mf=[0.002, 0.001])), init_prev=0.05)
        syph = ss.Disease(infection=ss.transmission('S -> I', beta=0.01), cure=ss.progression('I -> S', dur=ss.years(3)), init_prev=0.05)
        coinfection = [
            ss.modify(hiv.rel_trans, multiply=20.0, where=syph.I, name='hiv_trans'), # Both diseases have a state called I, so each condition remembers its own
            ss.modify(syph.rel_sus, multiply=5.0, where=hiv.I, name='syph_sus'),
        ]
        return ss.Sim(diseases=[hiv, syph], networks=ss.MFNet(), connectors=coinfection if coinfect else None, start=2000, stop=2020, verbose=0)
    sim = make().run()
    base = make(coinfect=False).run()
    assert list(sim.diseases.keys()) == ['hiv', 'syph'] # Named from their variables
    assert sim.results.hiv.n_I[-1] > base.results.hiv.n_I[-1]
    assert sim.results.syph.cum_infection[-1] > 1.1*base.results.syph.cum_infection[-1]
    assert 'where=syph.I' in sim.what_modifies('hiv.rel_trans', output=True)
    assert 'where=hiv.I' in sim.what_modifies('syph.rel_sus', output=True)
    return sim


def test_coinfection_compartments():
    sc.heading('Testing two diseases that modify each other, as compartments')
    def make(solver='ode', coinfect=True, interventions=None, strains=None):
        hiv = ss.Disease(infection=ss.transmission('S -> I', beta=0.15), death=ss.death('I ->', rate=0.05), init_prev=0.05)
        syph = ss.Disease(infection=ss.transmission('S -> I', beta=1.0), cure=ss.progression('I -> S', dur=ss.years(1)), init_prev=0.05)
        if strains:
            syph.stratify(strain=['A', 'B'], states=strains, split=[0.5, 0.5], adjust=dict(rel_trans=[1.0, 1.5]))
        coinfection = [
            ss.modify(hiv.rel_trans, multiply=3.0, where=syph.I, name='hiv_trans'),
            ss.modify(syph.rel_sus, multiply=2.0, where=hiv.I, name='syph_sus'),
        ]
        vx = ss.vaccinate(ss.Vaccine(efficacy=1.0), eligible=hiv.I, coverage=1.0, years=[2010], disease='syph') if interventions else None
        return ss.Sim(diseases=[hiv, syph], connectors=coinfection if coinfect else None, interventions=vx, solver=solver,
                      start=2000, stop=2030, dt=ss.years(0.1), total_pop=20_000, verbose=0)
    sims = sc.objdict({solver:make(solver).run() for solver in solvers})
    base = make(coinfect=False).run()

    # The states of the two diseases are tracked jointly, so they have the same people, and the results of each can be split by the other
    res = sims.ode.results
    assert 'hiv × syph: 2 × 2 = 4' in sims.ode.describe(output=True)
    assert np.allclose(res.hiv.n_S + res.hiv.n_I, res.syph.n_S + res.syph.n_I) # Deaths from HIV are deaths of people with syphilis too
    with_syph = res.hiv.n_I.by('syph').values[-1] / (res.hiv.n_S.by('syph').values[-1] + res.hiv.n_I.by('syph').values[-1])
    assert with_syph[1] > with_syph[0] # More HIV among people with syphilis
    assert res.hiv.n_I[-1] > 1.2*base.results.hiv.n_I[-1]
    assert res.syph.n_I[-1] > 2*base.results.syph.n_I[-1]
    for solver in ['binomial', 'abm']:
        for key in ['hiv', 'syph']:
            assert np.isclose(sims[solver].results[key].n_I[-1], res[key].n_I[-1], rtol=0.1)

    # A delivery of one disease's product to people in a state of the other
    sim = make(coinfect=False, interventions=True).run()
    res = sim.results
    ti = np.argmax(res.vx.delivered > 0)
    assert np.isclose(res.vx.delivered[ti], res.hiv.n_I[ti])

    # A dimension of some states of one disease only, e.g. a strain
    sim = make(strains=['I']).run()
    res = sim.results
    assert np.allclose(res.syph.n_I.by('strain').values.sum(axis=1), res.syph.n_I)
    assert np.allclose(res.hiv.n_S + res.hiv.n_I, res.syph.n_S + res.syph.n_I)
    with pytest.raises(ss.CodedError, match='E804'): # Not yet, since people born to the other disease start in the first state
        make(strains=['S', 'I']).init()
    return sims


def test_loop_order():
    sc.heading('Testing moving a module in the loop')
    def make():
        test = ss.Test(sensitivity=0.9, specificity=0.98)
        drug = ss.Treatment(efficacy=0.95)
        screen = ss.screen(test, coverage=0.2, interval=1, on_positive=ss.treat(drug))
        sim = ss.Sim(make_sir(), interventions=screen, n_agents=5000, stop=60, verbose=0, unit='day')
        return sim, screen
    sim, screen = make()
    sim.run()
    labels = sim.loop.to_df().label.tolist()
    assert labels.index('screen.step') < labels.index('sir.step') # By default, interventions come before transmission

    moved, screen = make()
    moved.init()
    moved.loop.insert(screen, label='sir.step') # So someone can be infected, tested, and treated in one step
    moved.run()
    df = moved.loop.to_df()
    labels = df.label.tolist()
    assert labels.index('screen.step') == labels.index('sir.step') + 1
    assert (df.label == 'screen.step').sum() == (df.label == 'sir.step').sum() # Moved, not added
    assert moved.results.screen.diagnosed.sum() > 0
    assert not np.array_equal(moved.results.sir.n_I, sim.results.sir.n_I)
    return moved


def test_newborns_and_events():
    sc.heading('Testing routine delivery to newborns, and deliveries on a transition that an intervention causes')
    pars = dict(stop=6*365, n_agents=20_000, **kw)
    ppl = ss.People(100)

    # Routine vaccination of children reaches those born or aging in, in a compartmental run too
    def make(solver):
        sir = make_sir(births=ss.birth('-> S', rate=ss.peryear(0.03)), deaths=ss.death(rate=ss.peryear(0.03))).stratify(age=[0, 5])
        kids = ss.vaccinate(ss.Vaccine(efficacy=0.9), eligible=ppl.age < 5, name='kids')
        return ss.Sim(sir, interventions=kids, solver=solver, **pars).run()
    sims = sc.objdict(ode=make('ode'), abm=make('abm'))
    for sim in sims.values():
        assert sim.results.kids.delivered[-365:].sum() > 0.5*50_000*0.03 # Still delivering in the last year, to newborns
    check_close(sims, lambda sim: sim.results.kids.delivered.sum())

    # A delivery on a transition sees the people that a treatment moved along it, as well as those the model did
    for solver in ['ode', 'abm']:
        sir = make_sir()
        cure = ss.treat(ss.Treatment(efficacy=1.0, transition='recovery'), eligible=sir.I, coverage=0.2, interval=1, name='cure')
        after = ss.screen(ss.Test(sensitivity=1.0), on=sir.recovery, name='after')
        sim = ss.Sim(sir, interventions=[after, cure], solver=solver, stop=150, n_agents=20_000, **kw).run() # The delivery that reads is first, so it sees them a step later
        res = sim.results
        assert np.isclose(res.after.delivered.sum(), res.sir.recovery[:-1].sum(), rtol=0.01) # Everyone who recovered, but on the last step
    return sims


def test_triggers():
    sc.heading('Testing interventions triggered by the state of the sim')
    pars = dict(stop=200, **kw)
    def make():
        return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.001, name='sir')

    # A function of the sim has no memory, so it switches on and off as the number infected crosses the threshold
    sir = make()
    distancing = ss.modify(sir.pars.beta, multiply=0.3, when=lambda sim: sim.results.sir.n_I[sim.ti - 1] > 1000)
    bare = ss.Sim(sir, interventions=distancing, **pars).run()
    base = ss.Sim(make(), **pars).run()
    assert bare.solver == 'ode'
    assert bare.results.sir.n_I.max() < 0.2*base.results.sir.n_I.max()

    # A policy with memory is a module that decides, and a modifier that applies the effect
    class Closure(ss.Intervention):
        """ Close for four weeks once the number infected passes a threshold """
        compartmental = True # It reads results and sets a flag, so it runs under any solver

        def __init__(self, threshold=1000, weeks=4, **kwargs):
            super().__init__(**kwargs)
            self.threshold = threshold
            self.weeks = weeks
            self.closed = False
            self.reopen = None
            self.days_closed = 0
            return

        def step(self):
            sim = self.sim
            if self.closed and sim.ti >= self.reopen:
                self.closed = False
            if not self.closed and sim.results.sir.n_I[sim.ti - 1] > self.threshold:
                self.closed = True
                self.reopen = sim.ti + 7*self.weeks
            self.days_closed += self.closed
            return

    sir = make()
    effect = ss.modify(sir.pars.beta, multiply=0.3, when=lambda sim: sim.interventions.closure.closed)
    held = ss.Sim(sir, interventions=[Closure(name='closure'), effect], **pars).run()
    assert held.solver == 'ode'
    assert held.interventions.closure.days_closed % 28 == 0 or held.interventions.closure.closed # Closed for four weeks at a time
    assert held.results.sir.n_I.max() < base.results.sir.n_I.max()
    assert np.isclose(sir.pars.beta.value, 0.3) # The base value is never overwritten
    return held



def test_sensitivity_by_state():
    sc.heading('Testing a test whose sensitivity depends on the state, and on age within a state')
    def make(**kwargs):
        S, E, I, R = ss.states(4)
        seir = ss.Disease(S >> E @ 0.3*I, E >> I @ ss.days(4), I >> R @ ss.days(8), init=dict(I=0.01), name='seir').stratify(age=[0, 15])
        test = ss.Test(sensitivity=dict(I=ss.by(age=[0.5, 0.9]), E=0.2), specificity=1.0)
        screening = ss.screen(test, coverage=0.5, interval=10, start=20)
        return ss.Sim(seir, interventions=screening, stop=60, **sc.mergedicts(dict(total_pop=50_000), kwargs))
    for solver,pars in [('ode', {}), ('abm', dict(n_agents=50_000))]:
        res = make(solver=solver, **kw, **pars).run().results
        ti = 30
        before = lambda name, **sel: res.seir[f'n_{name}'].select(**sel)[ti] if sel else res.seir[f'n_{name}'][ti]
        expected = 0.5*(0.5*before('I', age=0) + 0.9*before('I', age=1) + 0.2*before('E')) # Nobody else tests positive
        print(f'{solver}: {res.screening.diagnosed[ti]:.0f} diagnosed, expected {expected:.0f}')
        assert np.isclose(res.screening.diagnosed[ti], expected, rtol=0.05 if solver == 'ode' else 0.15)
    with pytest.raises(ss.CodedError, match='E505'):
        ss.Sim(make_sis(), interventions=ss.screen(ss.Test(sensitivity=dict(X=0.5)), interval=10), stop=20, **kw).run()
    return res


def test_trace():
    sc.heading('Testing contact tracing: delivery to the contacts of people something has happened to')
    pars = dict(stop=120, n_agents=5_000, solver='abm', rand_seed=1, **kw)
    def run(ring):
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.1), recovery=ss.progression('I -> R', dur=10), init_prev=0.002, name='sir')
        net = ss.StaticNet(graph=nx.erdos_renyi_graph, p=8/5_000, seed=True, name='net')
        vx = ss.Vaccine(efficacy=1.0)
        intvs = ss.vaccinate(vx, on=sir.infection, trace='net', name='ring') if ring else None
        return ss.Sim(sir, networks=net, interventions=intvs, **pars).run()
    base = run(False)
    sim = run(True)
    infected = sim.results.sir.infection.sum()
    print(f'Infected: {base.results.sir.infection.sum():.0f} without and {infected:.0f} with ring vaccination ({sim.results.ring.delivered.sum():.0f} vaccinated)')
    assert infected < 0.5*base.results.sir.infection.sum()

    # Everyone vaccinated is a contact of someone infected
    sir = sim.diseases.sir
    ever = (sir.I | sir.R).uids
    vaccinated = ss.uids(np.isfinite(sim.interventions.ring.product.ti_delivered.raw).nonzero()[0])
    assert len(vaccinated) and set(vaccinated) <= set(sim.networks.net.find_contacts(ever))

    # Only agents keep contacts, and it needs to know whose
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')
    with pytest.raises(ss.CodedError, match='E804'):
        ss.Sim(sir, interventions=ss.vaccinate(ss.Vaccine(), on=sir.infection, trace='net'), solver='ode', **kw).init()
    with pytest.raises(ValueError, match='on='):
        ss.Sim(sir, networks=ss.RandomNet(name='net'), interventions=ss.vaccinate(ss.Vaccine(), trace='net'), solver='abm', n_agents=1000, **kw).init()
    return sim


if __name__ == '__main__':
    sc.options(interactive=False)
    change = test_change()
    where  = test_modify_where()
    conds  = test_conditions()
    vx     = test_vaccinate()
    trans  = test_transfer()
    treat  = test_treat()
    screen = test_screen()
    sens   = test_sensitivity_by_state()
    msim   = test_deliver()
    coinf  = test_coinfection()
    coinfc = test_coinfection_compartments()
    loop   = test_loop_order()
    newb   = test_newborns_and_events()
    held   = test_triggers()
    trace  = test_trace()
