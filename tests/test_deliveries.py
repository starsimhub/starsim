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
    check_close(sims, lambda sim: sim.results.sir.infection.cum[-1])

    # Values instead of multipliers, and dates
    sir = make_sir()
    to = ss.change(sir.pars.beta, at=['2020-01-21', '2020-03-01'], to=[0.09, 0.3])
    dated = ss.Sim(sir, interventions=to, start='2020-01-01', stop='2020-04-30', total_pop=50_000, verbose=0).run()
    assert np.allclose(dated.results.sir.n_S, res.n_S)
    sir = make_sir()
    bare = ss.Sim(sir, interventions=ss.change(sir.pars.beta, at=[20, 60], to=[0.09, 0.3]), start='2020-01-01', stop='2020-04-30', total_pop=50_000, verbose=0).run()
    assert np.allclose(bare.results.sir.n_S, res.n_S) # In a sim with dates, a number is days since the start (not the year 20, which was taken as the first day)
    with pytest.raises(ss.TimeOutsideSim): # A time outside the sim, such as the year 2030 written as a number of days
        ss.Sim(sir, interventions=ss.change(sir.pars.beta, at=[2030], to=[0.09]), start='2020-01-01', stop='2020-04-30', verbose=0).init()
    with pytest.raises(ss.TimeOutsideSim): # At the stop, where no step starts, so it would never apply
        ss.Sim(sir, interventions=ss.change(sir.pars.beta, at=['2020-04-30'], to=[0.09]), start='2020-01-01', stop='2020-04-30', verbose=0).init()
    with pytest.raises(ValueError):
        ss.change(sir.pars.beta, at=[20, 60], by=[0.3])

    # It is a modifier, so stop= and when= work
    for extra in [dict(stop=0), dict(when=lambda sim: False)]:
        sir = make_sir()
        sim = ss.Sim(sir, interventions=ss.change(sir.pars.beta, at=[0], by=[0.3], **extra), solver='ode', stop=120, **kw).run()
        assert np.isclose(sim.results.sir.infection.cum[-1], base.results.sir.infection.cum[-1]) # Not 492 infections against 9,309

    # A modifier applies whatever form the parameter takes
    forms = dict(scalar=0.3, func=lambda sim: 0.3, timeseries=ss.timeseries({0: 0.3, 100: 0.3}), seasonal=ss.seasonal(mean=0.3, amplitude=0.2, peak=0))
    for solver in ['ode', 'abm']:
        for key,beta in forms.items():
            sir = ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')
            stopped = ss.modify(sir.pars.beta, multiply=0.0)
            sim = ss.Sim(sir, interventions=stopped, solver=solver, stop=50, **kw).run()
            assert sim.results.sir.infection.sum() == 0, f'{key} under {solver}'
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
    check_close(sims, lambda sim: sim.results.sir.infection.cum[-1])
    text = sims.ode.what_modifies('sir.rel_sus', output=True)
    assert 'risky' in text and 'multiply=3.0' in text and 'higher' in text
    assert 'faster' in sims.abm.what_modifies(sims.abm.diseases.sir.recovery, output=True)
    assert 'none' in sims.ode.what_modifies('sir.pars.beta', output=True)

    # A multiplier that is an expression of the states of another disease is worked out at each step, not once at the start
    def make_two(solver, protect=True, **kwargs):
        a = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> S', dur=10), init_prev=0.01, name='a')
        b = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> S', dur=10), init_prev=0.01, name='b')
        protection = ss.modify(a.rel_sus, multiply=1 - 0.8*b.I.frac) if protect else None
        return ss.Sim(diseases=[a, b], connectors=protection, solver=solver, **kwargs)
    two = run_each(make_two)
    base = make_two('ode', protect=False, stop=150, **kw).run()
    final = check_close(two, lambda sim: sim.results.a.n_I[-1], rtol=0.2) # The solvers agree (as closely as b's own levels do)
    assert final.ode < 0.8*base.results.a.n_I[-1] # And the protection grows as b spreads

    # A modifier of a parameter can depend on the group, but in a compartmental run not on the state
    sir = make_sir().stratify(risk=['lower', 'higher'])
    group = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.0, where=sir.risk == 'lower'), solver='ode', stop=50, **kw).run()
    assert np.isclose(group.results.sir.infection.by('risk').values[:, 0].sum(), 0)
    with pytest.raises(ss.CapabilityError):
        ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.5, where=sir.I), solver='ode', stop=50, **kw).run()

    # set= and add= are for parameters: on a per-person quantity or a rate they would replace each person's own value
    for op in ['set', 'add']:
        with pytest.raises(ValueError):
            ss.Sim(make_sir(), interventions=ss.modify('sir.rel_sus', **{op:1.0}), stop=20, **kw).init()

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
    assert isinstance(cond, ss.conditions.Cond)
    assert str(cond) == '((people.age < 15) & sir.S)'
    assert str((ss.ppl.age < 15) & sir.S) == str(cond) # The same condition with no people made
    assert str(ppl.age + 1 > 3) == str(ss.ppl.age + 1 > 3) and str(ppl.age.isin([1, 2])) == str(ss.ppl.age.isin([1, 2])) # Arithmetic and isin() are lazy too
    assert not hasattr(ss.People, 'age') # ss.People is the class only: the lazy references are ss.ppl
    with pytest.raises(TypeError):
        bool(cond)

    # The same condition selects agents or groups
    def make(eligible=cond, **kwargs):
        vaccine = ss.Vaccine(efficacy=0.9)
        kids = ss.vaccinate(vaccine, where=eligible, at=[5])
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
    with pytest.raises(ss.CapabilityError):
        make(eligible=ppl.age < 10, solver='ode', stop=20, **kw).run()

    # The model's own age groups, compared with an age, are compared with their edges
    for solver in ['ode', 'abm']:
        delivered = [make(eligible=e & sir.S, solver=solver, stop=20, **kw).run().results.kids.delivered.sum() for e in [sir.age < 15, sir.age == '0-14']]
        assert np.isclose(delivered[0], delivered[1]) and np.isclose(delivered[0], sims[solver].results.kids.delivered.sum()) # Not everyone
    with pytest.raises(ValueError):
        make(eligible=sir.age < 10, solver='abm', stop=20, **kw).run()
    return sims


def test_vaccinate():
    sc.heading('Testing vaccination: campaigns, routine delivery, waning, and capacity')
    base = ss.Sim(make_sir(), solver='ode', stop=150, **kw).run()

    # A campaign
    def campaign(**kwargs):
        sir = make_sir()
        vaccine = ss.Vaccine(efficacy=0.9, cost=2.0)
        rounds = ss.vaccinate(vaccine, coverage=0.6, at=[5])
        return ss.Sim(sir, interventions=rounds, **kwargs)
    sims = run_each(campaign)
    for sim in sims.values():
        res = sim.results.rounds
        assert np.isclose(res.delivered.sum(), 30_000, rtol=0.02) and np.isclose(res.coverage[5], 0.6, atol=0.02)
        assert np.isclose(res.cost.sum(), 2*res.delivered.sum())
        assert sim.results.sir.infection.cum[-1] < 0.5*base.results.sir.infection.cum[-1]
        assert [strat.name for strat in sim.strats] == ['has_vaccine'] # Holding the vaccine is a property of people
    assert sims.ode.hash().model == base.hash().model # The scenario is not part of the model
    assert 'has_vaccine' in sims.ode.describe(output=True)
    rounds = ss.vaccinate(efficacy=0.9, cost=2.0, coverage=0.6, at=[5]) # With a vaccine of its own
    own = ss.Sim(make_sir(), interventions=rounds, solver='ode', stop=150, **kw).run()
    assert np.isclose(own.results.sir.infection.cum[-1], sims.ode.results.sir.infection.cum[-1]) # The same as with the vaccine given
    assert [strat.name for strat in own.strats] == ['has_rounds_vaccine']

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

    # Routine delivery by age is given as people reach the age, not to everyone already older at the start
    for solver in ['ode', 'abm']:
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ ss.perday(0.1)*I, I >> R @ ss.days(8), ss.birth('-> S', rate=ss.peryear(0.03)), init=dict(I=0.001), name='sir').stratify(age=[0, 0.75, 5])
        mcv1 = ss.vaccinate(ss.Vaccine(efficacy=0.85), coverage=0.83, where=ss.People(1).age >= 0.75, name='mcv1') # The spec's routine at 9 months
        delivered = ss.Sim(sir, interventions=mcv1, start='2021-01-01', stop='2022-01-01', solver=solver, **kw).run().results.mcv1.delivered.values
        assert delivered[:7].sum() < 0.001*kw['total_pop'] and delivered.sum() > 0 # Not 41,000 of 50,000 in the first week

    # Routine delivery to everyone of a vaccine that wanes is still one dose each
    for solver in ['ode', 'abm']:
        sim = ss.Sim(make_sir(), interventions=ss.vaccinate(ss.Vaccine(efficacy=0.9, waning=ss.days(10)), start=0, name='everyone'), solver=solver, stop=100, **kw).run()
        assert np.isclose(sim.results.everyone.delivered.sum(), 49_500, rtol=0.02) # Not 520,000: once each, not again every time the protection is lost

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
        assert np.allclose(delivered[10:40], 500, rtol=0.15) and delivered[:10].sum() == 0 and delivered[40:].sum() == 0 # The binomial solver draws the number reached, so is a little under the capacity on some days
    with pytest.raises(ValueError): # A bare number for routine delivery has no unit of time (it was per unit time, while in a campaign it is per round)
        ss.Sim(make_sir(), interventions=ss.vaccinate(ss.Vaccine(efficacy=0.9), capacity=100, start=10), **kw).init()

    # Errors
    with pytest.raises(ss.UnknownEffect):
        ss.Sim(make_sir(), interventions=ss.vaccinate(ss.Vaccine(efficacy=dict(aquisition=0.5))), **kw).init()
    with pytest.raises(TypeError):
        ss.vaccinate(ss.Diagnostic())
    a, b = [ss.Disease(infection=ss.transmission('S -> I', beta=0.3), init_prev=0.01, name=name) for name in 'ab']
    vx = ss.Vaccine(efficacy=0.9)
    with pytest.raises(ss.UnknownDisease): # A product acts on one disease, so delivered for a second it would protect against nothing
        ss.Sim([a, b], interventions=[ss.vaccinate(vx, at=[5], disease='a', name='d1'), ss.vaccinate(vx, at=[5], disease='b', name='d2')], **kw).init()

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

    # A capacity per unit time, in rounds: what accrues over the interval between them
    def rounds_cap(**kwargs):
        vaccine = ss.Vaccine(efficacy=0.9)
        rounds = ss.vaccinate(vaccine, coverage=1.0, interval=ss.days(10), capacity=ss.perday(100))
        return ss.Sim(make_sir(), interventions=rounds, **kwargs)
    sims = run_each(rounds_cap, stop=30)
    for sim in sims.values():
        delivered = sim.results.rounds.delivered
        assert list(np.flatnonzero(delivered)) == [0, 10, 20] and np.allclose(delivered[[0, 10, 20]], 1000)

    # A typed probability as coverage: over the interval between rounds
    def typed_cov(**kwargs):
        rounds = ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=ss.probperyear(0.5), interval=ss.days(91))
        return ss.Sim(make_sir(), interventions=rounds, **kwargs)
    sims = run_each(typed_cov, stop=100)
    for sim in sims.values():
        assert np.isclose(sim.results.rounds.coverage[0], 1 - 0.5**(91/365), rtol=0.05)
    return sims


def test_transfer():
    sc.heading('Testing vaccination as a transfer along a stratification')
    def make(**kwargs):
        sir = make_sir().stratify(vaccinated=[False, True], adjust={'infection': [1.0, 0.15]}) # A boolean dimension starts with everyone False
        rollout = ss.Delivery(move=dict(vaccinated=True), coverage=0.7, at=[5, 30], where=~sir.vaccinated)
        return ss.Sim(sir, interventions=rollout, **kwargs)
    sims = run_each(make)
    for sim in sims.values():
        delivered = sim.results.rollout.delivered
        assert np.allclose(delivered[[5, 30]], [35_000, 10_500], rtol=0.05) # 70% of everyone, then 70% of those left (not of the half that started unvaccinated)
    check_close(sims, lambda sim: sim.results.sir.infection.cum[-1], rtol=0.3) # What is left after the campaign depends on how far the epidemic had got, which varies by about 10% between agent-based runs, and agents are about 10% below the ODE at a step of a day
    with pytest.raises(TypeError): # ss.transfer() is movement at a rate, never a delivery
        ss.transfer(vaccinated=True, coverage=0.7, at=[5])

    # From one level to another, given as an arrow
    sir = make_sir().stratify(risk=['low', 'high'], split=[1, 0])
    sim = ss.Sim(sir, interventions=ss.Delivery(move=dict(risk='low -> high'), coverage=0.4, at=[5], name='riskier'), solver='ode', stop=10, **kw).run()
    assert np.isclose(sim.results.riskier.delivered[5], 0.4*kw['total_pop'], rtol=0.01)
    return sims


def test_treat():
    sc.heading('Testing treatment: curing, and changing rates')
    base = ss.Sim(make_sis(), solver='ode', stop=150, **kw).run()

    # A treatment that cures
    def cure(**kwargs):
        sis = make_sis()
        drug = ss.Treatment(cure=0.9)
        rounds = ss.treat(drug, coverage=0.3, interval=5, start=30)
        return ss.Sim(sis, interventions=rounds, **kwargs)
    sims = run_each(cure)
    for solver,sim in sims.items():
        res = sim.results
        assert res.sis.n_I[-1] < 0.9*base.results.sis.n_I[-1]
        assert np.isclose(res.rounds.delivered[30], 0.3*res.sis.n_I[30], rtol=0.15 if solver == 'abm' else 0.07) # By default, the people who are infected at the time (as agents, those infected after the step's exits, and one run)
    check_close(sims, lambda sim: sim.results.rounds.delivered.sum(), rtol=0.12) # As agents, about 9% fewer are infected (a difference of order dt)

    # A treatment that changes rates, with protection that fades by a function of time (in a compartmental run, levels of time since delivery)
    def make_sird():
        return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), death=ss.death('I ->', rate=0.02), init_prev=0.01, name='sird')
    def rates(waning=None, **kwargs):
        sird = make_sird()
        art = ss.Treatment(efficacy=dict(transmission=0.5, mortality=0.9), waning=waning)
        program = ss.treat(art, where=sird.I, coverage=0.8)
        return ss.Sim(sird, interventions=program, **kwargs)
    untreated = ss.Sim(make_sird(), solver='ode', stop=150, **kw).run()
    sims = run_each(rates)
    for sim in sims.values():
        assert sim.results.sird.death.sum() < 0.6*untreated.results.sird.death.sum()
    check_close(sims, lambda sim: sim.results.program.delivered.sum()/sim.results.sird.infection.cum[-1]) # Most of the people infected, whenever that was
    fades = run_each(rates, waning=lambda t: np.exp(-t/5))
    for solver,sim in fades.items():
        assert sim.results.sird.death.sum() > 1.2*sims[solver].results.sird.death.sum()
    check_close(fades, lambda sim: sim.results.sird.death.sum())
    assert rates(waning=lambda t: np.exp(-t/5), approx=False, **kw).init().solver == 'abm' # The levels are an approximation

    with pytest.raises(ss.UnknownEffect):
        ss.Sim(make_sird(), interventions=ss.treat(ss.Treatment(cure=0.5, transition='cure')), **kw).init()

    # A cure takes people out of every infected state, not only the sources of the transition named (before, those in Y stayed)
    for solver in ['ode', 'abm']:
        S, A, Y = ss.State(), ss.State(init=0.5), ss.State(init=0.5)
        sti = ss.Disease(S=S, A=A, Y=Y, infection=S >> A @ ss.perday(1e-9)*(A + Y), symp=A >> Y @ ss.days(1e5), clear_a=A >> S @ ss.days(1000), clear_y=Y >> S @ ss.days(1000), name='sti')
        tx = ss.treat(ss.Treatment(transition='clear_a'), at=[5], name='tx')
        res = ss.Sim(sti, interventions=tx, unit='day', stop=10, solver=solver, n_agents=2000, verbose=0).run().results.sti
        assert res.n_A[-1] == 0 and res.n_Y[-1] == 0 # Was 995 left in Y
    with pytest.raises(ss.UnknownEffect): # Its recoveries go to Y and S, so it does not guess which (to= says)
        ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.1), carrier=ss.progression('I -> C', dur=5), recovery=ss.progression('I -> R', dur=10), name='sic'), interventions=ss.treat(ss.Treatment(cure=0.5)), **kw).init()
    with pytest.raises(TypeError): # A value of the sim's time, not of the time since delivery
        ss.Treatment(efficacy=dict(transmission=0.5), waning=ss.timeseries({0: 1, 10: 0.5}))

    # A treatment held is for the infected by default, like one that cures; and one can do both
    for solver in ['ode', 'abm']:
        sird = make_sird()
        both = ss.treat(ss.Treatment(cure=0.5, efficacy=dict(mortality=0.9)), coverage=0.5, interval=5, name='both')
        sim = ss.Sim(sird, interventions=both, solver=solver, stop=150, **kw).run()
        res = sim.results
        assert np.isclose(res.both.delivered[0], 0.5*res.sird.n_I[0], rtol=0.1 if solver == 'ode' else 0.3) # Not to everyone (it was 10,000 of 10,000); as agents, half of about 100, after the step's exits
        assert res.sird.death.sum() < 0.6*untreated.results.sird.death.sum()
    for bad in [lambda: ss.Treatment(efficacy=0.9), lambda: ss.Vaccine()]:
        with pytest.raises(ValueError):
            bad()
    return sims


def test_screen():
    sc.heading('Testing screening, follow-ups, and event-triggered delivery')

    # Screen in rounds, and treat most of those who test positive
    def rounds(**kwargs):
        sis = make_sis()
        test = ss.Diagnostic(sensitivity=0.9, specificity=0.95)
        drug = ss.Treatment(cure=0.95)
        screening = ss.screen(test, coverage=0.4, interval=7, start=30)
        treatment = ss.treat(drug, who=screening.positive, coverage=0.8)
        return ss.Sim(sis, interventions=[treatment, screening], **kwargs) # In any order: the follow-up runs after the test
    sims = run_each(rounds)
    for solver,sim in sims.items():
        res = sim.results
        rtol = 0.05 # Under every solver the screening sees the people at the start of the step, before the disease's transitions in it (as agents it saw them after the exits, and was 8% low)
        ti = 37
        n_I = res.sis.n_I[ti] # At the time
        expected = 0.4*(0.9*n_I + 0.05*(50_000 - n_I)) # True positives and false positives
        assert np.isclose(res.screening.delivered[ti], 20_000, rtol=0.03)
        assert np.isclose(res.screening.positive[ti], expected, rtol=rtol)
        assert np.isclose(res.treatment.delivered[ti], 0.8*res.screening.positive[ti], rtol=0.05) # The follow-up is a step of its own, on the step of the test
        assert np.isclose(res.screening.positive[ti] + res.screening.negative[ti], res.screening.delivered[ti])
        assert np.isclose(res.treatment.cured[ti] + res.treatment.failed[ti] + res.treatment.unneeded[ti], res.treatment.delivered[ti])
        assert 'test' not in res and 'drug' not in res # Products have no results of their own
        assert np.isclose(sim.summarize()['screening_coverage'], 0.4, rtol=0.03) # Over the rounds, not averaged with the steps between them (was 0.06)
    check_close(sims, lambda sim: sim.results.screening.positive.sum())

    # Test half of new infections, and cure those found
    def events(**kwargs):
        sir = make_sir()
        test = ss.Diagnostic(sensitivity=0.8)
        drug = ss.Treatment(cure=1.0)
        symptomatic = ss.screen(test, who=sir.infection, coverage=0.5)
        treatment = ss.treat(drug, who=symptomatic.positive)
        return ss.Sim(sir, interventions=[symptomatic, treatment], **kwargs)
    sims = run_each(events)
    for sim in sims.values():
        res = sim.results
        assert np.isclose(res.symptomatic.delivered.sum(), 0.5*res.sir.infection[:-1].sum(), rtol=0.05)
        assert np.isclose(res.symptomatic.positive.sum(), 0.8*res.symptomatic.delivered.sum(), rtol=0.08) # As agents, a few have already recovered by the time they are tested
    check_close(sims, lambda sim: sim.results.sir.infection.cum[-1])

    # A test with a delay: the follow-ups happen when the result arrives (in a compartmental run, people wait in levels of one step each)
    def delay(**kwargs):
        sis = make_sis()
        test = ss.Diagnostic(sensitivity=0.9, specificity=0.9, delay=ss.days(3))
        drug = ss.Treatment(cure=1.0)
        slow = ss.screen(test, coverage=0.5, at=[40])
        cure = ss.treat(drug, who=slow.positive)
        protect = ss.vaccinate(ss.Vaccine(efficacy=0.5), who=slow.negative)
        return ss.Sim(sis, interventions=[slow, cure, protect], **kwargs)
    sims = run_each(delay, stop=60)
    for sim in sims.values():
        res = sim.results
        assert list(np.flatnonzero(res.slow.delivered)) == [40] and list(np.flatnonzero(res.slow.positive)) == [43]
        assert np.isclose(res.cure.delivered[43], res.slow.positive[43])
        assert np.isclose(res.slow.positive[43] + res.protect.delivered[43], res.slow.delivered[40], rtol=0.01) # Everyone tested has one result or the other
    check_close(sims, lambda sim: sim.results.slow.positive.sum()) # As agents, the prevalence of an SIS model is about 6% lower at a step of a day

    # Someone waiting for a result is not tested again, so a test repeated more often than its delay still returns results
    for solver in ['ode', 'abm']:
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0), recovery=ss.progression('I -> R', dur=ss.days(1e6)), init_prev=1.0, name='sir')
        daily = ss.screen(ss.Diagnostic(sensitivity=1.0, specificity=1.0, delay=ss.days(3)), interval=1, name='daily')
        res = ss.Sim(sir, interventions=daily, solver=solver, total_pop=100, **(dict(n_agents=100) if solver == 'abm' else {}), unit='day', stop=20, verbose=0).run().results.daily
        assert np.isclose(res.delivered.sum(), 700) and np.isclose(res.positive.sum(), 600, rtol=1e-3) # Each person every 3 days, the last result after the end; not 2,000 tests and 0 positive

    # Nested follow-ups are removed, with a message that says how to write them as steps; and ss.Test is now ss.Diagnostic
    for key in ['on_positive', 'on_negative']:
        with pytest.raises(TypeError, match='step of its own'):
            ss.screen(ss.Diagnostic(), interval=7, **{key:ss.treat(ss.Treatment())})
    with pytest.raises(AttributeError, match='Diagnostic'):
        ss.Test()
    return sims


def test_cascade():
    sc.heading('Testing a cascade: steps triggered by each other\'s events, and its table')

    # A union of events fires once per person, written as | or as a list; an intersection is those in both
    for solver in ['ode', 'abm']:
        out = {}
        for key,who in dict(union=lambda a,b: a.positive | b.positive, list=lambda a,b: [a.positive, b.positive], both=lambda a,b: [a.positive, b.positive]).items():
            a = ss.screen(ss.Diagnostic(sensitivity=1.0, name='dx_a'), coverage=0.5, at=[20], name='a')
            b = ss.screen(ss.Diagnostic(sensitivity=1.0, name='dx_b'), coverage=0.5, at=[20], name='b')
            tx = ss.treat(ss.Treatment(cure=1.0), who=who(a, b), operator='all' if key == 'both' else 'any', name='tx')
            res = ss.Sim(make_sir(), interventions=[tx, a, b], solver=solver, stop=30, **kw).run().results # In any order: each step after its triggers
            out[key] = res.tx.delivered[20]/res.sir.n_I[20]
        print(f'{solver}: {out}')
        assert np.isclose(out['union'], 0.75, atol=0.03) and out['list'] == out['union'] # Half of the infected tested by each, independently: 75% by either, once each (not 100%)
        assert np.isclose(out['both'], 0.25, atol=0.03)

    # Several steps on the same event, in the step of the test, under agents and the ODE (each finds the people where the others moved them)
    for solver in ['ode', 'abm']:
        s = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.9), coverage=0.5, at=[20], name='s')
        cure = ss.treat(ss.Treatment(cure=0.5), who=s.positive, name='cure')
        v1 = ss.vaccinate(ss.Vaccine(efficacy=0.5, name='vx1'), who=s.positive, name='v1')
        v2 = ss.vaccinate(ss.Vaccine(efficacy=0.5, name='vx2'), who=s.positive, name='v2')
        res = ss.Sim(make_sir(), interventions=[s, cure, v1, v2], solver=solver, stop=30, **kw).run().results
        n = res.s.positive[20]
        assert n > 0 and np.allclose([res.cure.delivered[20], res.v1.delivered[20], res.v2.delivered[20]], n) # Not fewer for the later steps
        assert np.isclose(res.cure.cured[20] + res.cure.failed[20] + res.cure.unneeded[20], n)

    # A cascade runs as its steps would as a list, and adds a table of who each step reached and lost
    def make(as_cascade=True, **kwargs):
        sis = make_sis()
        rapid = ss.Diagnostic(sensitivity=0.8, specificity=0.95)
        pcr = ss.Diagnostic(sensitivity=0.95, specificity=0.99, delay=ss.days(2))
        screening = ss.screen(who=rapid.since >= ss.days(28), what=rapid, coverage=0.1, interval=ss.days(7))
        confirm = ss.screen(who=screening.positive, what=pcr, coverage=0.9)
        treatment = ss.treat(who=confirm.positive, what=ss.Treatment(cure=0.95), capacity=ss.perday(50)) # Its treatment is called treatment_treatment
        care = ss.cascade(screening, confirm, treatment)
        return ss.Sim(make_sis(), interventions=care if as_cascade else [treatment, confirm, screening], **kwargs)
    for solver in ['ode', 'abm']:
        sim = make(solver=solver, stop=120, **kw).run()
        plain = make(as_cascade=False, solver=solver, stop=120, **kw).run()
        assert np.array_equal(sim.results.sis.n_I, plain.results.sis.n_I) # Identical
        df = sim.results.care.to_df()
        res = sim.results
        assert list(df.step) == ['screening', 'confirm', 'treatment'] # Named from their variables
        assert np.allclose(df.reached + df.lost_coverage + df.lost_capacity, df.eligible)
        assert df.lost_capacity.iloc[2] > 0 and np.isclose(df.eligible.iloc[1], res.screening.positive.sum()) # Each step's eligible are its trigger's outcome
        assert np.isclose(df.eligible.iloc[2], res.confirm.positive.sum(), rtol=1e-6)
    text = sim.describe(output=True)
    assert 'care (cascade of 3 steps)' in text and 'on screening.positive' in text

    # The steps are registered for a subclass of ss.Sim that passes the modules in pars
    class SubSim(ss.Sim):
        def __init__(self, interventions=None, **kwargs):
            super().__init__(pars=dict(interventions=interventions), **kwargs)
    rapid = ss.Diagnostic(sensitivity=0.8, specificity=0.95)
    pcr = ss.Diagnostic(sensitivity=0.95, specificity=0.99, delay=ss.days(2))
    screening = ss.screen(who=rapid.since >= ss.days(28), what=rapid, coverage=0.1, interval=ss.days(7))
    confirm = ss.screen(who=screening.positive, what=pcr, coverage=0.9)
    treatment = ss.treat(who=confirm.positive, what=ss.Treatment(cure=0.95), capacity=ss.perday(50))
    sub = SubSim(interventions=ss.cascade(screening, confirm, treatment, name='care'), diseases=make_sis(), solver='abm', stop=120, **kw).run()
    assert 'confirm' in sub.interventions and np.array_equal(sub.results.sis.n_I, sim.results.sis.n_I)
    assert sim.results.care.plot() is not None

    # The errors: a cycle of triggers, an event outside the sim, an event | a condition, a misspelled outcome, and capacity by stratum (TL47)
    a = ss.screen(ss.Diagnostic(), interval=7, name='a')
    b = ss.screen(ss.Diagnostic(), who=a.positive, name='b')
    a.on = b.positive # A retest triggered by the test it follows
    with pytest.raises(ValueError, match='cycle'):
        ss.Sim(make_sis(), interventions=[a, b], **kw)
    a = ss.screen(ss.Diagnostic(), interval=7, name='a')
    with pytest.raises(ValueError, match='not delivered in the sim'):
        ss.Sim(make_sis(), interventions=ss.treat(ss.Treatment(), who=a.positive), **kw)
    with pytest.raises(TypeError, match='use &'):
        ss.treat(ss.Treatment(), who=a.positive | (ss.ppl.age > 5))
    with pytest.raises(TypeError, match='use &'):
        ss.treat(ss.Treatment(), who=[a.positive, ss.ppl.age > 5])
    with pytest.raises(AttributeError, match='positive, negative'):
        a.postive
    with pytest.raises(ValueError, match='capacity is a total'):
        sis = make_sis().stratify(risk=['lo', 'hi'])
        ss.Sim(sis, interventions=ss.treat(ss.Treatment(), who=sis.infection, capacity=ss.by(risk=[1, 2])), **kw).init()
    return sim


def test_memory():
    sc.heading('Testing the memory of events: .ever, .count, .since, and count_events=')

    # Nobody screened twice within five years; the time since as levels against agents; first-time positives by count
    def make(**kwargs):
        sis = ss.Disease(infection=ss.transmission('S -> I', beta=ss.peryear(2)), recovery=ss.progression('I -> S', dur=ss.years(1)), init_prev=0.05, name='sis')
        cytology = ss.Diagnostic(sensitivity=0.8, specificity=0.95)
        screening = ss.screen(cytology, who=cytology.since >= ss.years(5), coverage=0.3, interval=ss.years(1), count_events=True)
        retest = ss.screen(ss.Diagnostic(sensitivity=0.9, name='pcr'), who=screening.positive & (screening.positive.count == 1), coverage=0.5)
        return ss.Sim(sis, interventions=ss.cascade(screening, retest, name='care'), start=2000, stop=2030, total_pop=20_000, verbose=0, **kwargs)
    sims = sc.objdict(abm=make(n_agents=20_000).run(), ode=make(solver='ode').run())
    df = sims.abm.interventions.screening.events_df()
    tested = df[df.event == 'screening'].sort_values(['uid', 'time'])
    assert len(tested) == sims.abm.results.screening.delivered.sum() # The log has one row per event per person
    assert tested.groupby('uid').time.diff().dropna().min() >= np.timedelta64(5*365, 'D')
    scr = sims.abm.interventions.screening
    assert (scr.count_positive.raw > 0).sum() == df[df.event == 'screening.positive'].uid.nunique() # Positives over the run are those ever positive
    assert 'since_cytology' in sims.ode.describe(output=True)
    for key,rtol in [('screening', 0.05), ('retest', 0.08)]: # Retests as agents are about 6% below the ODE, whose memory of events is approximated (independent within each state and group)
        assert np.isclose(sims.ode.results[key].delivered.sum(), sims.abm.results[key].delivered.sum(), rtol=rtol)

    # An ss.Event of a module written as code (agents): one visit, both tests on the same attendees, and memory of a test at delivery
    sis = ss.Disease(infection=ss.transmission('S -> I', beta=ss.peryear(0.5)), recovery=ss.progression('I -> S', dur=ss.years(2)), init_prev=0.1, name='sis')
    preg = ss.LegacyPregnancy(fertility_rate=50)
    anc_visit = ss.visit(who=preg.conception, coverage=0.85)
    anc_a = ss.screen(ss.Diagnostic(sensitivity=0.9, name='rdt'), who=anc_visit)
    anc_b = ss.screen(ss.Diagnostic(sensitivity=0.8, name='rpr'), who=anc_visit)
    newborn = ss.treat(ss.Treatment(cure=1.0, name='tx'), who=preg.delivery & anc_a.positive.ever)
    sim = ss.Sim(sis, demographics=preg, interventions=[anc_visit, anc_a, anc_b, newborn], n_agents=5000, start=2000, stop=2010, dt=1/12, verbose=0).run()
    res = sim.results
    assert sim.solver == 'abm' and res.anc_visit.delivered.sum() > 0 and res.newborn.delivered.sum() > 0
    assert np.array_equal(res.anc_a.delivered.values, res.anc_visit.delivered.values) and np.array_equal(res.anc_b.delivered.values, res.anc_visit.delivered.values)

    # A transfer triggers a delivery, as agents (TL43); a state that only a delivery moves people into is not where they start (TL38)
    S, I = ss.states('S I')
    d = ss.Disease(name='d', infection=S >> I @ 0.1)
    d.stratify(sym=[False, True], onset=ss.transfer(sym=True, rate=0.2, states=['I']))
    sim = ss.Sim(diseases=d, interventions=ss.screen(ss.Diagnostic(), who=d.onset, coverage=0.8, name='scr'), n_agents=2000, start=2000, stop=2005, dt=0.25, verbose=0).run()
    assert sim.solver == 'abm' and sim.results.scr.delivered.sum() > 0
    S, M, H = ss.states(3)
    dis = ss.Disease(S >> M @ ss.peryear(0.5), M >> S @ ss.days(10), H >> S @ ss.days(5), name='dis')
    sim = ss.Sim(dis, interventions=ss.Delivery(move='M -> H', who=M, interval=ss.days(1)), n_agents=1000, stop=ss.days(30), dt=ss.days(1), verbose=0).init()
    assert sim.diseases.dis.graph.first.name == 'S' # Was H, so everyone started in hospital
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
        campaign = ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.6, at=[5])
        return ss.Sim(sir, interventions=campaign if vaccinate else None, solver='binomial', stop=100, **kw)
    msim = ss.parallel(baseline=scenario(False), vaccine=scenario(True), n_runs=3)
    diff = msim.diff('baseline').vaccine.sir.infection.cum
    assert diff.values.shape == (3, 101)
    assert np.all(diff.values[:, -1] < 0) # Fewer infections in every run
    fig = msim.plot('sir.n_I') # The median and an interval for each scenario
    assert len(fig.axes[0].lines) == 2
    return msim


def test_coinfection():
    sc.heading('Testing two diseases that modify each other')
    def make(coinfect=True):
        hiv = ss.Disease(infection=ss.transmission('S -> I', beta=dict(mf=ss.peract(m2f=0.002, f2m=0.001))), init_prev=0.05)
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
    assert sim.results.syph.infection.cum[-1] > base.results.syph.infection.cum[-1] # About 13% more on average, but from 3% to 26% between single runs
    assert 'where=syph.I' in sim.what_modifies('hiv.rel_trans', output=True)
    assert 'where=hiv.I' in sim.what_modifies('syph.rel_sus', output=True)
    return sim


def test_coinfection_compartments():
    sc.heading('Testing two diseases that modify each other, as compartments')
    def make(solver='ode', coinfect=True, interventions=None, strains=None):
        hiv = ss.Disease(infection=ss.transmission('S -> I', beta=0.15), death=ss.death('I ->', rate=0.05), init_prev=0.05)
        syph = ss.Disease(infection=ss.transmission('S -> I', beta=1.0), cure=ss.progression('I -> S', dur=ss.years(1)), init_prev=0.05)
        if strains:
            syph.stratify(strain=['A', 'B'], kind='strain', states=strains, split=[0.5, 0.5], adjust=dict(rel_trans=[1.0, 1.5]))
        coinfection = [
            ss.modify(hiv.rel_trans, multiply=3.0, where=syph.I, name='hiv_trans'),
            ss.modify(syph.rel_sus, multiply=2.0, where=hiv.I, name='syph_sus'),
        ]
        vx = ss.vaccinate(ss.Vaccine(efficacy=1.0), where=hiv.I, coverage=1.0, at=[2010], disease='syph') if interventions else None
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
            assert np.isclose(sims[solver].results[key].n_I[-1], res[key].n_I[-1], rtol=0.15) # As agents, syphilis is about 5% above the ODE on average, and varies by about 3% between runs
    stepped = make('binomial').run(until=2030) # The stochastic run takes the diseases together, by one program, whole or step by step
    assert sims.binomial.kernel_only() and np.array_equal(stepped.results.syph.n_I, sims.binomial.results.syph.n_I)

    # Each disease's moves are drawn as they would be for it alone, so two diseases with dwell times in stages, tracked jointly but not changing each other, peak as they do alone (drawn as one compartment's competing exits, they would be 7% and 31% too high at daily steps)
    def staged(crossed=True, which=None, seed=0):
        seir = lambda name, dur: ss.Disease(infection=ss.transmission('S -> E', beta=0.3, source='I'), latent=ss.progression('E -> I', dur=ss.lognorm_ex(5, 2)), recovery=ss.progression('I -> R', dur=ss.lognorm_ex(dur, 3)), init_prev=0.01, name=name)
        a, b = seir('a', 10), seir('b', 8)
        same = ss.modify(a.rel_sus, multiply=1.0, where=b.I, name='same') if crossed else None
        return ss.Sim(diseases=[a, b] if which is None else dict(a=a, b=b)[which], connectors=same, solver='binomial', unit='day', stop=120, total_pop=100_000, rand_seed=seed, verbose=0).run()
    for key in ['a', 'b']:
        crossed = np.mean([staged(seed=seed).results[key].n_I.max() for seed in range(4)])
        alone = np.mean([staged(crossed=False, which=key, seed=seed).results[key].n_I.max() for seed in range(4)])
        assert np.isclose(crossed, alone, rtol=0.02)

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
    with pytest.raises(ss.CapabilityError): # Not yet, since people born to the other disease start in the first state
        make(strains=['S', 'I']).init()
    return sims


def test_loop_order():
    sc.heading('Testing moving a module in the loop')
    def make():
        test = ss.Diagnostic(sensitivity=0.9, specificity=0.98)
        drug = ss.Treatment(cure=0.95)
        screen = ss.screen(test, coverage=0.2, interval=1)
        treatment = ss.treat(drug, who=screen.positive)
        sim = ss.Sim(make_sir(), interventions=[screen, treatment], n_agents=5000, stop=60, verbose=0, unit='day')
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
    assert moved.results.screen.positive.sum() > 0
    assert not np.array_equal(moved.results.sir.n_I, sim.results.sir.n_I)
    return moved


def test_newborns_and_events():
    sc.heading('Testing routine delivery to newborns, and deliveries on a transition that an intervention causes')
    pars = dict(stop=6*365, n_agents=20_000, **kw)
    ppl = ss.People(100)

    # Routine vaccination of children reaches those born or aging in, in a compartmental run too
    def make(solver):
        sir = make_sir(births=ss.birth('-> S', rate=ss.peryear(0.03)), deaths=ss.death(rate=ss.peryear(0.03))).stratify(age=[0, 5])
        kids = ss.vaccinate(ss.Vaccine(efficacy=0.9), where=ppl.age < 5, name='kids')
        return ss.Sim(sir, interventions=kids, solver=solver, **pars).run()
    sims = sc.objdict(ode=make('ode'), abm=make('abm'))
    for sim in sims.values():
        assert sim.results.kids.delivered[-365:].sum() > 0.5*50_000*0.03 # Still delivering in the last year, to newborns
    check_close(sims, lambda sim: sim.results.kids.delivered.sum())

    # A delivery on a transition sees the people that a treatment moved along it, as well as those the model did
    for solver in ['ode', 'abm']:
        sir = make_sir()
        cure = ss.treat(ss.Treatment(cure=1.0, transition='recovery'), where=sir.I, coverage=0.2, interval=1, name='cure')
        after = ss.screen(ss.Diagnostic(sensitivity=1.0), who=sir.recovery, name='after')
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
            if not self.closed and sim.results.sir.n_I[sim.ti] > self.threshold: # During a step, sim.ti is its start, the last time point with results (it was [sim.ti - 1] when code saw the step's end)
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

    # Or hold=: a condition that holds on one step keeps the modifier on for ten, as start= and stop= would
    sir = make()
    once = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.3, when=lambda sim: sim.ti == 50, hold=10), **pars).run()
    sir = make()
    timed = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.3, start=50, stop=60), **pars).run()
    assert np.allclose(once.results.sir.n_I, timed.results.sir.n_I)
    return held


def test_move_leaky_at_age():
    sc.heading('Testing a one-off move between states, an all-or-nothing vaccine, and routine doses at an age')
    def make():
        S, I, R, V = ss.states(4)
        return ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(8), V >> S @ ss.days(100), init=dict(I=10), name='sir')
    for solver in ['ode', 'abm']:
        sim = ss.Sim(make(), interventions=ss.vaccinate(move='S -> V', at=[5], coverage=0.5, name='sia'), unit='day', stop=10, solver=solver, n_agents=10_000, verbose=0).run()
        res, moved = sim.results.sir, sim.results.sia.delivered[5]
        assert np.isclose(moved, 0.5*res.n_S[5], rtol=0.05) and np.isclose(res.n_V.max(), moved, rtol=0.02) # Half of those in S move to V, at once
    with pytest.raises(ValueError):
        ss.vaccinate(ss.Vaccine(efficacy=0.9), move='S -> V')

    # All or nothing: 85% of those vaccinated fully protected and 15% not at all, so with R0 = 8 an epidemic among the 15% infects about 5% of everyone; a leaky vaccine leaves everyone 15% susceptible, and about 31% are infected
    def final(leaky, solver):
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=1.0), recovery=ss.progression('I -> R', dur=8), init=dict(I=10), name='sir')
        vx = ss.vaccinate(ss.Vaccine(efficacy=0.85, leaky=leaky), at=[1], coverage=1.0, name='vx')
        sim = ss.Sim(sir, interventions=vx, unit='day', stop=300, solver=solver, n_agents=20_000, verbose=0).run()
        assert np.isclose(sim.results.vx.delivered[1], sim.results.sir.n_S[1], rtol=0.01) # Everyone received it
        return sim.results.sir.n_R[-1]/sim.results.sir.n_S[0]
    for solver in ['ode', 'abm']:
        aon, leaky = final(False, solver), final(True, solver)
        assert aon < 0.1 and leaky > 3*aon # As agents at a daily step, leaky gives 17% rather than 31%, which is of order dt

    # Routine at nine months: nobody older at the start, and children as they reach it
    ppl = ss.People(10_000, age_data=np.array([np.arange(80), np.exp(-0.035*np.arange(80))]).T)
    S, I, R = ss.states(3)
    sir = ss.Disease(S >> I @ ss.perday(0.1)*I, I >> R @ ss.days(8), ss.birth('-> S', rate=ss.peryear(0.03)), init=dict(I=0.001), name='sir')
    ri = ss.vaccinate(ss.Vaccine(efficacy=0.85), coverage=0.833, at_age=ss.months(9), name='routine')
    sim = ss.Sim(sir, people=ppl, interventions=ri, start='2021-01-01', stop='2022-01-01', unit='day', verbose=0).run()
    delivered = sim.results.routine.delivered.values
    assert delivered[:7].sum() < 0.01*len(ppl) and delivered.sum() > 0
    return sim


def test_coverage_by_group():
    sc.heading('Testing that the coverage of several groups is what was delivered over who was eligible in them')
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.2), recovery=ss.progression('I -> R', dur=10), init=dict(I=10), name='sir')
    sir.stratify(age=[0, 5, 15])
    vx = ss.vaccinate(ss.Vaccine(efficacy=0.9), at=[5], coverage=ss.by(age=[0.8, 0.2, 0.5]), name='vx')
    res = ss.Sim(sir, interventions=vx, unit='day', stop=20, solver='ode', verbose=0).run().results.vx
    young = ['0-4', '5-14']
    assert np.allclose(res.coverage.by('age').values[5], [0.8, 0.2, 0.5])
    assert np.isclose(res.coverage.select(age=young)[5], res.delivered.select(age=young)[5]/res.eligible.select(age=young)[5]) # Not 0.8 + 0.2
    return res


def test_hold_by_place():
    sc.heading('Testing a rule that each place applies on its own, on its incidence, with a hold')
    def run(solver, by='patch', hold=ss.days(14), npi=True):
        beta = ss.par(0.3)
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', rate=ss.perday(0.1)), init=dict(I=ss.by(patch=[10, 0, 0])), name='sir')
        sir.stratify(patch=['a', 'b', 'c'], mobility=ss.Mobility(matrix=0.01*(1 - np.eye(3)))) # The other places are infected later
        cond = sir.infection.incidence(over=ss.days(7), by=by) > 0.02 # A week's infections in each place, as a share of its people
        rule = ss.modify(beta, multiply=0.3, where=cond, hold=hold) if npi else None
        sim = ss.Sim(sir, interventions=rule, unit='day', stop=200, solver=solver, n_agents=20_000, verbose=0, rand_seed=1).run()
        return sim.results.sir.n_I.strata.max(axis=0)
    for solver in ['ode', 'abm']:
        base, local = run(solver, npi=False), run(solver)
        assert np.all(local < 0.2*base) # Each place cuts its own epidemic
    local = run('ode')
    assert not np.allclose(run('ode', by=None), local) # Unlike one rule for the whole population
    assert np.all(run('ode', hold=None) > local) # And the hold keeps it down longer
    return local



def test_sensitivity_by_state():
    sc.heading('Testing a test whose sensitivity depends on the state, and on age within a state')
    def make(**kwargs):
        S, E, I, R = ss.states(4)
        seir = ss.Disease(S >> E @ 0.3*I, E >> I @ ss.days(4), I >> R @ ss.days(8), init=dict(I=0.01), name='seir').stratify(age=[0, 15])
        test = ss.Diagnostic(sensitivity=dict(I=ss.by(age=[0.5, 0.9]), E=0.2), specificity=1.0)
        screening = ss.screen(test, coverage=0.5, interval=10, start=20)
        return ss.Sim(seir, interventions=screening, stop=60, **sc.mergedicts(dict(total_pop=50_000), kwargs))
    for solver,pars in [('ode', {}), ('abm', dict(n_agents=50_000))]:
        res = make(solver=solver, **kw, **pars).run().results
        ti = 30
        before = lambda name, **sel: res.seir[f'n_{name}'].select(**sel)[ti] if sel else res.seir[f'n_{name}'][ti]
        expected = 0.5*(0.5*before('I', age=0) + 0.9*before('I', age=1) + 0.2*before('E')) # Nobody else tests positive
        print(f'{solver}: {res.screening.positive[ti]:.0f} positive, expected {expected:.0f}')
        assert np.isclose(res.screening.positive[ti], expected, rtol=0.05 if solver == 'ode' else 0.15)
    with pytest.raises(ss.UnknownEffect):
        ss.Sim(make_sis(), interventions=ss.screen(ss.Diagnostic(sensitivity=dict(X=0.5)), interval=10), stop=20, **kw).run()
    return res


def test_treat_any_transition():
    sc.heading('Testing a treatment that moves people along a transition out of a state that is not infected')
    for solver,pars in [('ode', {}), ('abm', dict(n_agents=10_000))]:
        S, I, R, P = ss.states(4)
        sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), protect=ss.Transition(S, P, rate=ss.perday(0)), init=dict(I=0.01), name='sir')
        shield = ss.treat(ss.Treatment(transition='protect'), where=sir.S, coverage=0.5, at=[ss.days(0)])
        res = ss.Sim(sir, interventions=shield, stop=100, solver=solver, **kw, **pars).run().results
        print(f'{solver}: {res.sir.protect.sum():.0f} protected')
        assert np.isclose(res.sir.protect.sum(), 0.5*0.99*kw['total_pop'], rtol=0.05)
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
    assert sim.solvers(output=True).count('module: ring') == 1 # Its reason for agents, not that and a trial on groups as well

    # Everyone vaccinated is a contact of someone infected
    sir = sim.diseases.sir
    ever = (sir.I | sir.R).uids
    vaccinated = ss.uids(np.isfinite(sim.interventions.ring.product.ti_delivered.raw).nonzero()[0])
    assert len(vaccinated) and set(vaccinated) <= set(sim.networks.net.find_contacts(ever))

    # Only agents keep contacts, and it needs to know whose
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')
    with pytest.raises(ss.CapabilityError):
        ss.Sim(sir, interventions=ss.vaccinate(ss.Vaccine(efficacy=0.9), on=sir.infection, trace='net'), solver='ode', **kw).init()
    with pytest.raises(ss.CapabilityError, match='solver="gillespie"'): # The solver asked for, not the ODE that estimates its cost
        ss.Sim(sir, interventions=ss.vaccinate(ss.Vaccine(efficacy=0.9), on=sir.infection, trace='net'), solver='gillespie', **kw).init()
    with pytest.raises(ValueError, match='an event in who='):
        ss.Sim(sir, networks=ss.RandomNet(name='net'), interventions=ss.vaccinate(ss.Vaccine(efficacy=0.9), trace='net'), solver='abm', n_agents=1000, **kw).init()
    with pytest.raises(TypeError): # on= is an event; the people in a state are where=sir.I, interval=1
        ss.vaccinate(ss.Vaccine(efficacy=0.9), on=sir.I)
    return sim


def test_branch_modifiers():
    sc.heading('Testing modifiers on a branch chosen with p=: they change its probability, and the other branches take the rest')
    def make(solver, intv=None, where=None):
        dis = ss.Disease(
            infection = ss.transmission('S -> I', beta=0.3, source='I'),
            severe    = ss.progression('I -> H', p=0.2, dur=5, where=where),
            mild      = ss.progression('I -> R', p=0.8, dur=10),
            discharge = ss.progression('H -> R', dur=7),
            init_prev = 0.01, name='dis',
        )
        intvs = intv(dis) if intv is not None else None
        pars = dict(n_agents=20_000) if solver == 'abm' else {}
        sim = ss.Sim(dis, interventions=intvs, solver=solver, stop=200, **kw, **pars)
        if where is not None:
            sim.stratify(age=[0, 50])
        return sim.run().results.dis
    share = lambda res: res.severe.sum()/(res.severe.sum() + res.mild.sum())
    halve = lambda dis: ss.modify(dis.severe, multiply=0.5, name='halve')
    vaccine = lambda dis: ss.vaccinate(ss.Vaccine(efficacy=dict(severe=0.75)), at=0, name='vx')
    for solver in solvers:
        base, halved, vx = make(solver), make(solver, halve), make(solver, vaccine)
        print(f'{solver}: severe share {share(base):.3f}, {share(halved):.3f} halved, {share(vx):.3f} vaccinated')
        assert np.isclose(share(base), 0.2, rtol=0.05)
        assert np.isclose(share(halved), 0.1, rtol=0.05)
        assert np.isclose(share(vx), 0.05, rtol=0.1)

    # A parameter in a branch probability changed by ss.change(), under every solver: 0.8 go to B after the change (not 0.2 as compartments)
    for multi in [True, False]:
        sB = ss.par(0.2, name='sB')
        trs = dict(leave=ss.Transition('A', ['B', 'C'], dur=ss.years(1), p=[sB, 1 - sB])) if multi else dict(a_to_B=ss.Transition('A', 'B', dur=ss.years(1), p=sB), a_to_C=ss.Transition('A', 'C', dur=ss.years(1), p=1 - sB))
        for solver in solvers:
            d = ss.Disease(**trs, back=ss.Transition(['B', 'C'], 'A', rate=ss.peryear(4)), name='d')
            size = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
            r = ss.Sim(d, interventions=ss.change('d.pars.sB', at=2005, to=0.8), start=2000, stop=2012, dt=ss.months(1), solver=solver, verbose=0, **size).run().results.d
            assert np.isclose(r.n_B[-1]/(r.n_B[-1] + r.n_C[-1]), 0.8, atol=0.05), solver

    # An efficacy on one destination of a transition with several halves its probability, without changing the dwell time
    for solver in ['ode', 'abm']:
        P, C, R = ss.states('P C R')
        hpv = ss.Disease(P, C, R, precin=ss.Transition(P, [C, R], p=[0.2, 0.8], dur=ss.years(2)), name='hpv')
        txvx = ss.vaccinate(ss.Vaccine(efficacy={'precin.C': 0.5}), coverage=0.5, at=[2000], name='txvx')
        size = dict(n_agents=20_000) if solver == 'abm' else dict(total_pop=20_000)
        res = ss.Sim(hpv, interventions=txvx, start=2000, stop=2012, dt=ss.years(0.25), solver=solver, verbose=0, **size).run().results.hpv
        assert np.isclose(res.n_C[-1]/20_000, 0.15, atol=0.01) # Half at 0.1, half at 0.2
        assert np.isclose(res.n_P[8]/20_000, np.exp(-1), rtol=0.1) # Left P after two years on average, as without it

    # A branch for some people only: the others take the other branch
    ppl = ss.People(1)
    for solver in ['ode', 'abm']:
        res = make(solver, where=ppl.age >= 50)
        print(f'{solver}: severe share {share(res):.3f} with where=')
        assert 0 < share(res) < 0.15
    return res


def test_network_contacts():
    sc.heading('Testing modifiers on the contacts of a network')
    def make(solver, intv=None, age=False):
        sir = make_sir()
        sir.pars.beta = 0.03
        pars = dict(n_agents=20_000) if solver == 'abm' else {}
        sim = ss.Sim(sir, networks=ss.MixingPool(n_contacts=10, name='school'), interventions=intv, solver=solver, stop=150, **kw, **pars)
        if age:
            sim.stratify(age=[0, 20])
        return sim.run().results.sir.infection.sum()
    halve = lambda: ss.modify('school.pars.n_contacts', multiply=0.5, start=20)
    for solver in solvers:
        base, halved, set5 = make(solver), make(solver, halve()), make(solver, ss.modify('school.pars.n_contacts', set=5, start=20))
        young = make(solver, ss.modify('school.pars.n_contacts', multiply=0.5, start=20, where=ss.People(1).age < 20), age=True)
        print(f'{solver}: {base:.0f} infections, {halved:.0f} with half the contacts from day 20, {young:.0f} if only for those under 20')
        assert np.isclose(halved, set5)
        assert halved < young < base
        assert np.isclose(halved, 0.8*base, rtol=0.1) # Agents differ from the ODE by an amount of order dt

    # Only a network's contacts can be targeted
    with pytest.raises(ss.UnmodifiableTarget):
        make('ode', ss.modify('school.pars.dur', multiply=0.5))
    return halved


def test_effective():
    sc.heading('Testing base and effective values')
    sir = make_sir()
    protect = ss.modify(sir.rel_sus, multiply=0.4, where=sir.R, name='protect')
    sim = ss.Sim(sir, interventions=protect, solver='abm', n_agents=2000, stop=40, **kw).run()
    arr = sim.diseases.sir.rel_sus
    R = sim.diseases.sir.R.uids
    assert np.allclose(arr.effective[R], 0.4*arr[R]) # The base is untouched by the modifier
    assert 'protect' in sim.what_modifies(arr, output=True)
    ode = ss.Sim(make_sir(), interventions=ss.vaccinate(efficacy=0.9, at=[5]), solver='ode', stop=40, **kw).run()
    with pytest.raises(ss.CapabilityError): # There are no agents to have values
        ode.diseases.sir.rel_sus.effective
    assert 'multiply=0.1 ' in ode.what_modifies('sir.rel_sus', output=True) # Not 0.09999999999999998
    return sim


def test_whole_run():
    sc.heading('Testing deliveries in a whole run in the kernels')
    def make(**kwargs):
        sir = make_sir().stratify(age=[0, 15, 65]).stratify(protected=[False, True], split=[1, 0], adjust={'infection': [1.0, 0.5]})
        routine = ss.Vaccine(efficacy=0.9, waning=ss.days(60))
        offer = ss.vaccinate(routine, where=sir.S, coverage=ss.by(age=[0.6, 0.3, 0.0]), start=10) # Each person offered it once
        campaign = ss.Vaccine(efficacy=0.7)
        rounds = ss.vaccinate(campaign, coverage=0.5, at=[20, 40])
        rollout = ss.Delivery(move=dict(protected=True), coverage=0.1, at=[30]) # Along a stratification
        return ss.Sim(sir, interventions=[offer, rounds, rollout], stop=100, **kw, **kwargs)
    whole = make(solver='ode').init()
    assert whole.kernel_only(), whole.kernel_reason()
    whole.run()
    stepped = make(solver='ode').init()
    stepped.kernel_only = lambda: False # As the loop takes it, one step at a time
    stepped.run()
    for mod,key in [('sir', 'n_S'), ('sir', 'n_I'), ('offer', 'delivered'), ('offer', 'eligible'), ('rounds', 'coverage'), ('rollout', 'delivered')]:
        a, b = whole.results[mod][key], stepped.results[mod][key]
        assert np.allclose(a.values, b.values, rtol=1e-10, atol=1e-8), f'{mod}.{key} differs'
        if getattr(b, 'strata', None) is not None:
            assert np.allclose(a.strata, b.strata, rtol=1e-10, atol=1e-8)

    # Many runs in one call: the same as each run on its own, deliveries included
    msim = make(solver='binomial').run(n_runs=4)
    single = make(solver='binomial', rand_seed=3).run()
    assert np.array_equal(msim.results.offer.delivered.values[2], single.results.offer.delivered.values)
    assert np.array_equal(msim.results.sir.n_I.values[2], single.results.sir.n_I.values)

    # And in JAX, with gradients
    if ss.gradients.available():
        jax = make(solver='ode').init()
        jax.jax = True # A whole run in JAX, as ss.parallel() takes many
        jax.run()
        assert np.allclose(jax.results.sir.n_I.values, whole.results.sir.n_I.values, rtol=1e-10)
        assert np.allclose(jax.results.offer.delivered.values, whole.results.offer.delivered.values, rtol=1e-10)
        grad = make(solver='ode').gradient('n_I', pars='beta').beta.sir.n_I
        h = 1e-5
        up, down = [make(solver='ode').new(pars=dict(beta=0.3 + d)).run().results.sir.n_I.values for d in [h, -h]]
        assert np.allclose(grad, (up - down)/(2*h), rtol=1e-4, atol=1e-3*np.abs(grad).max())
    return whole


if __name__ == '__main__':
    sc.options(interactive=False)
    change = test_change()
    where  = test_modify_where()
    conds  = test_conditions()
    vx     = test_vaccinate()
    trans  = test_transfer()
    treat  = test_treat()
    screen = test_screen()
    casc   = test_cascade()
    memory = test_memory()
    sens   = test_sensitivity_by_state()
    msim   = test_deliver()
    coinf  = test_coinfection()
    coinfc = test_coinfection_compartments()
    loop   = test_loop_order()
    newb   = test_newborns_and_events()
    held   = test_triggers()
    moves  = test_move_leaky_at_age()
    cover  = test_coverage_by_group()
    places = test_hold_by_place()
    anytr  = test_treat_any_transition()
    trace  = test_trace()
    branch = test_branch_modifiers()
    nets   = test_network_contacts()
    effective = test_effective()
    whole  = test_whole_run()
