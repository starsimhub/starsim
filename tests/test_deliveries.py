"""
Test interventions on declared models: modifiers, conditions, products, and their delivery
"""

# %% Imports and settings
import os
import tempfile
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl
from corpus import make_sir, make_sis, check_close, check_agents
import corpus

kw = dict(verbose=0, unit='day', total_pop=50_000)


def run_each(make, stop=150, **kwargs):
    """ Run the sim that `make` returns with each solver """
    return corpus.run_each(make, stop=stop, **kw, **kwargs)


# %% Modifiers

def test_change():
    sc.heading('Testing changes at given times: ss.modify() of a time series, of each form of parameter, and by date')
    def make(**kwargs):
        sir = make_sir()
        lockdown = ss.modify(sir.pars.beta, multiply=ss.timeseries({20: 0.3, 60: 1.0}), when=ss.schedule(start=20))
        return ss.Sim(sir, interventions=lockdown, **kwargs)
    sims = run_each(make, stop=120)
    base = ss.Sim(make_sir(), solver='ode', stop=120, **kw).run()
    res, res0 = sims.ode.results.sir, base.results.sir
    assert np.allclose(res.S[:20], res0.S[:20]) # The same until it starts, at step 20
    assert res.infection[20:60].sum() < 0.5*res0.infection[20:60].sum()
    assert res.infection[60:].sum() > res0.infection[60:].sum() # The epidemic resumes when the multiplier goes back to 1
    check_close(sims, lambda sim: sim.results.sir.infection.cum[-1])

    # Values instead of multipliers, by date; a start after the stop never applies, so a sim can be run to an earlier stop
    dated = dict(start='2020-01-01', stop='2020-04-30', total_pop=50_000, verbose=0)
    sir = make_sir()
    to = ss.modify(sir.pars.beta, set=ss.timeseries({'2020-01-21': 0.09, '2020-03-01': 0.3}), when=ss.schedule(start='2020-01-21'))
    assert np.allclose(ss.Sim(sir, interventions=to, **dated).run().results.sir.S, res.S)
    late = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, set=0.09, when=ss.schedule(start=2030)), **dated).init()
    assert any('never applies' in note for note in late.notes)

    # until= a duration: a time that holds on one step keeps the modifier on for ten, as a window would
    sir = make_sir()
    once = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.3, when=ss.schedule([50]), until=10), solver='ode', stop=100, **kw).run()
    sir = make_sir()
    timed = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.3, when=ss.schedule(start=50, stop=60)), solver='ode', stop=100, **kw).run()
    assert np.allclose(once.results.sir.I, timed.results.sir.I)

    # A modifier applies whatever form the parameter takes
    forms = dict(scalar=0.3, func=lambda sim: 0.3, timeseries=ss.timeseries({0: 0.3, 100: 0.3}), seasonal=ss.seasonal(mean=0.3, amplitude=0.2, peak=0))
    for key,beta in forms.items():
        sir = make_sir(beta=beta)
        sim = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.0), solver='ode', stop=50, **kw).run()
        assert sim.results.sir.infection.sum() == 0, key
    return sims


def test_triggers():
    sc.heading('Testing modifiers triggered by the state of the sim: a function, a policy with memory, and until= a lower level')
    pars = dict(stop=200, **kw)
    def make():
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.001, name='sir')

    # A function of the sim has no memory, so it switches on and off as the number infected crosses the threshold
    sir = make()
    bare = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.3, who=lambda sim: sim.results.sir.I[sim.ti] > 1000), **pars).run()
    base = ss.Sim(make(), **pars).run()
    assert isinstance(bare.solver, ss.solvers.ode) and bare.results.sir.I.max() < 0.2*base.results.sir.I.max()

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

        def step(self):
            sim = self.sim
            if self.closed and sim.ti >= self.reopen:
                self.closed = False
            if not self.closed and sim.results.sir.I[sim.ti] > self.threshold: # During a step, sim.ti is its own index, and its stocks (the state at its start) are recorded already
                self.closed = True
                self.reopen = sim.ti + 7*self.weeks
            self.days_closed += self.closed

    sir = make()
    effect = ss.modify(sir.pars.beta, multiply=0.3, who=lambda sim: sim.interventions.closure.closed)
    held = ss.Sim(sir, interventions=[Closure(name='closure'), effect], **pars).run()
    assert isinstance(held.solver, ss.solvers.ode)
    assert held.interventions.closure.days_closed % 28 == 0 or held.interventions.closure.closed # Closed for four weeks at a time
    assert held.results.sir.I.max() < base.results.sir.I.max()
    assert np.isclose(sir.pars.beta.value, 0.3) # The base value is never overwritten

    # until= keeps a modifier on from when= until a lower level, under each kind of solver
    def run(solver, pop=1e6, until=True):
        sir = make()
        until = dict(until=sir.I.frac <= 0.05) if until else {}
        measure = ss.modify(sir.pars.beta, multiply=0.6, when=sir.I.frac >= 0.1, **until)
        return ss.Sim(sir, interventions=measure, total_pop=pop, dur=ss.days(200), solver=solver, verbose=0).run().results.sir.infection.values.sum()
    assert run('ode') < run('ode', until=False) # It stays on longer than the bare threshold
    assert run('abm', pop=20_000) < 20_000
    return held


def test_targets():
    sc.heading('Testing what modifiers apply to: an inflow, and base and effective values')
    for solver in ['ode', 'abm']:
        Sv = ss.State(people=False)
        mosq = ss.Disease(ss.State('S'), Sv, emergence=ss.unborn >> Sv @ ss.perday(100), mortality=Sv >> ss.dead @ ss.days(10), init=dict(Sv=1000), name='mosq')
        sim = ss.Sim(mosq, interventions=ss.modify(mosq.emergence, multiply=0.5, when=ss.schedule(start=10)), solver=solver, n_agents=1000, total_pop=1000, unit='day', stop=100, verbose=0).run()
        assert np.isclose(sim.results.mosq.Sv[-1], 500, rtol=0.05), solver

    # Base and effective values
    sir = make_sir()
    protect = ss.modify(sir.rel_sus, multiply=0.4, who=sir.R, name='protect')
    sim = ss.Sim(sir, interventions=protect, solver='abm', n_agents=2000, stop=40, **kw).run()
    arr = sim.diseases.sir.rel_sus
    R = sim.diseases.sir.R.uids
    assert np.allclose(sim.diseases.sir.current_value('rel_sus', R), 0.4*arr[R]) # The base is untouched by the modifier
    assert 'protect' in sim.what_modifies(arr)
    ode = ss.Sim(make_sir(), solver='ode', stop=40, **kw).init()
    with pytest.raises(ss.CapabilityError): # There are no agents to have values
        ode.diseases.sir.current_value('rel_sus')
    return sim


# %% Conditions and who is reached

def test_conditions():
    sc.heading('Testing conditions')
    ppl = ss.People(100)
    sir = make_sir().stratify(age=[0, 15, 65])
    cond = (ppl.age < 15) & sir.S
    assert isinstance(cond, ss.conditions.Cond) and str((ss.person.age < 15) & sir.S) == str(cond) # The same condition with no people made
    with pytest.raises(TypeError):
        bool(cond)

    # The same condition selects agents or groups
    def make(eligible=cond, **kwargs):
        kids = ss.vaccinate(ss.Vaccine(efficacy=0.9), who=eligible, when=[5])
        return ss.Sim(sir, interventions=kids, **kwargs)
    sims = run_each(make, stop=20)
    for sim in sims.values():
        n_S = sim.results.sir.S.by('age').values[5, 0]
        assert np.isclose(sim.results.kids.delivered[5], n_S, rtol=0.02) # Every susceptible child at time 5, and nobody else, in the step from it
        assert np.allclose(sim.results.kids.delivered.by('age').values[5], [sim.results.kids.delivered[5], 0, 0]) # By stratum: all in the youngest group
    check_close(sims, lambda sim: sim.results.kids.delivered.sum(), rtol=0.05)

    # A function of the sim does the same; the model's own age groups are compared with their edges
    func = make(eligible=lambda sim: sim.people.age < 15, solver='ode', stop=20, **kw).run()
    assert func.results.kids.delivered[5] > sims.ode.results.kids.delivered[5] # Including the children who are not susceptible
    edges = make(eligible=(sir.age == '0-14') & sir.S, solver='ode', stop=20, **kw).run()
    assert np.isclose(edges.results.kids.delivered.sum(), sims.ode.results.kids.delivered.sum())

    # An age that is not the edge of a group cannot be used on groups, so the model runs as agents unless told otherwise
    assert isinstance(make(eligible=ppl.age < 10, stop=20, **kw).init().solver, ss.solvers.abm)
    with pytest.raises(ss.CapabilityError):
        make(eligible=ppl.age < 10, solver='ode', stop=20, **kw).run()
    return sims


def test_paths():
    sc.heading('Testing states by their paths in who=, where=, and move=, the same as the states, and the note on seeding beside the default')
    def run(make, key='interventions', **kw):
        S, E, I, R = ss.states('S E I R')
        d = ss.Disease(infection=S >> E @ ss.perday(0.4)*I, onset=E >> I @ ss.days(3), recovery=I >> R @ ss.days(7), name='seir', **kw)
        return ss.Sim(d, **{key: make(d)}, total_pop=5000, dur=ss.days(60), verbose=0, solver='ode').run()
    R = lambda sim: sim.results.seir.R[-1]
    kw = dict(init_n=dict(I=10))
    assert R(run(lambda d: ss.treat(cure=1.0, who=d.I, rate=ss.perday(0.1)), **kw)) == R(run(lambda d: ss.treat(cure=1.0, who='seir.I', rate=ss.perday(0.1)), **kw))
    moves = [R(run(lambda d: ss.Event(move(d), when=ss.days(10), p=0.3), **kw)) for move in [lambda d: d.S >> d.R, lambda d: 'seir.S >> seir.R', lambda d: 'S >> R']]
    assert moves[0] == moves[1] == moves[2]
    seeding = lambda d: ss.Event(d.S >> d.I, when=ss.days(10), p=0.001)
    note = lambda sim: any('give the disease init=dict(I=0)' in note for note in sim.notes)
    sim = run(seeding)
    assert note(sim) and sim.results.seir.E[0] > 0 # The default seeding (into E, where infection puts people), with a note
    sim = run(seeding, init=dict(I=0))
    assert not note(sim) and sim.results.seir.E[0] == sim.results.seir.I[0] == 0
    return


def test_transfer():
    sc.heading('Testing vaccination as a transfer along a stratification')
    def make(**kwargs):
        sir = make_sir().stratify(vaccinated=[False, True]) # A boolean dimension starts with everyone False
        rollout = ss.Event(dict(vaccinated=True), who=~sir.vaccinated, when=[5, 30], p=0.7)
        return ss.Sim(sir, interventions=rollout, connectors=ss.modify(sir.infection, multiply=ss.by(vaccinated=[1.0, 0.15])), **kwargs)
    sims = run_each(make)
    for sim in sims.values():
        assert np.allclose(sim.results.rollout.delivered[[5, 30]], [35_000, 10_500], rtol=0.05) # 70% of everyone, then 70% of those left
    check_close(sims, lambda sim: sim.results.sir.infection.cum[-1], rtol=0.35) # What is left after the campaign depends on how far the epidemic had got, which varies by about 10% between agent-based runs, and agents are about 10% below the ODE at a step of a day

    # From one level to another, given as an arrow
    sir = make_sir().stratify(risk={'low': 1, 'high': 0})
    sim = ss.Sim(sir, interventions=ss.Event(dict(risk=('low', 'high')), when=[5], p=0.4, name='riskier'), solver='ode', stop=10, **kw).run()
    assert np.isclose(sim.results.riskier.delivered[5], 0.4*kw['total_pop'], rtol=0.01)

    # The disease is the one whose dimension it moves people along, not the one a condition in who= names
    S2, I2 = ss.states('S I')
    sis = ss.Disease(infection=S2 >> I2 @ 0.3*I2, recovery=I2 >> S2 @ 0.1, init_prev=0.01, name='sis')
    ev = ss.Event(dict(risk=('low', 'high')), who=I2, when=[5], p=0.5, name='ev')
    sim = ss.Sim(make_sir().stratify(risk=['low', 'high']), sis, interventions=ev, n_agents=200, stop=10, verbose=0).init()
    assert sim.interventions.ev.disease == 'sir'
    return sims


def test_deliver():
    sc.heading('Testing delivery by a function, triggers, yearly rounds on weekly steps, and coverage by group')

    # A function that says who is eligible, as any condition in who= does: UIDs, as agents
    chosen = ss.vaccinate(ss.Vaccine(efficacy=0.9), who=lambda sim: sim.people.auids[:10_000], when=5, capacity=8000)
    sim = ss.Sim(make_sir(), interventions=chosen, solver='abm', stop=20, **kw).run()
    assert np.isclose(sim.results.chosen.delivered.sum(), 8000) # Capped by the capacity

    # A trigger: vaccinate while there are many infections
    def triggered(**kwargs):
        many = lambda sim: sim.results.sir.I[sim.ti] > 2000 # At the start of the step
        response = ss.vaccinate(ss.Vaccine(efficacy=0.9), who=many, when=ss.schedule(every=1), coverage=0.2)
        return ss.Sim(make_sir(), interventions=response, **kwargs)
    for sim in run_each(triggered).values():
        given = np.flatnonzero(sim.results.response.delivered)
        assert len(given) and np.all(sim.results.sir.I[given] > 2000)

    # Rounds a year apart on weekly steps fall on the first step of each year
    yearly = ss.vaccinate(ss.Vaccine(efficacy=0.9), when=ss.schedule(every=ss.years(1)), coverage=0.2, name='yearly')
    sim = ss.Sim(make_sir(), interventions=yearly, start='2000-01-01', stop='2010-01-01', dt=ss.weeks(1), solver='ode', verbose=0).init()
    starts = [sim.t.tvec[ti] for ti in sorted(sim.interventions.yearly.round_tis)]
    assert [d.year for d in starts] == list(range(2000, 2010)) and all(d.month == 1 and d.day <= 7 for d in starts)

    # The coverage of several groups is what was delivered over who was eligible in them
    sir = make_sir(beta=0.2).stratify(age=[0, 5, 15])
    vx = ss.vaccinate(ss.Vaccine(efficacy=0.9), when=[5], coverage=ss.by(age=[0.8, 0.2, 0.5]), name='vx')
    res = ss.Sim(sir, interventions=vx, unit='day', stop=20, solver='ode', verbose=0).run().results.vx
    young = ['0-4', '5-14']
    assert np.allclose(res.coverage.by('age').values[5], [0.8, 0.2, 0.5])
    assert np.isclose(res.coverage.select(age=young)[5], res.delivered.select(age=young)[5]/res.eligible.select(age=young)[5]) # Not 0.8 + 0.2
    assert np.isclose(res.eligible.summarize(), res.eligible.values.sum()) # The offers made over the run
    return sim


def test_age_and_birth():
    sc.heading('Testing delivery as people reach an age and at birth, as compartments and agents; and the disease of where=')
    def make(solver, strata=[0, 0.75, 5], **kwargs):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(S, I, R, infection=S >> I @ ss.peryear(0.5)*I, recovery=I >> R @ ss.days(8), births=ss.birth(S, rate=ss.peryear(0.03)), init=dict(I=0.001), name='sir')
        if strata is not None:
            sir.stratify(age=strata)
        mcv1 = ss.vaccinate(efficacy=0.85, who=ss.at_age(ss.months(9)), coverage=0.8, name='mcv1') # On aging into the group that starts at 9 months
        bcg = ss.vaccinate(efficacy=0.5, when='sir.births', coverage=0.6, name='bcg') # A transition by its path
        return ss.Sim(sir, interventions=[mcv1, bcg], start=2000, stop=2004, dt=ss.weeks(1), solver=solver, total_pop=20_000, verbose=0, rand_seed=2, **kwargs)
    sims = {solver:make(solver, **(dict(n_agents=20_000) if solver == 'abm' else {})).run() for solver in ['ode', 'abm']}
    for key in ['mcv1', 'bcg']:
        ode, abm = [sim.results[key].delivered.sum() for sim in sims.values()]
        assert np.isclose(ode, abm, rtol=0.1), f'{key}: {ode:n} as compartments, {abm:n} as agents'
    for sim in sims.values(): # At birth: the share of the newborns
        assert np.isclose(sim.results.bcg.delivered.sum(), 0.6*sim.results.sir.births.sum(), rtol=0.1)
    with pytest.raises(ss.CapabilityError): # at_age= never switches to agents: it needs an age group that starts at the age
        make('auto', strata=[0, 1, 5]).init()

    # With two diseases, a condition on one's own age groups says which disease a delivery is for
    measles, flu = [make_sir(name=name).stratify(age=[0, 15, 100]) for name in ['measles', 'flu']]
    sim = ss.Sim(diseases=[measles, flu], interventions=ss.vaccinate(efficacy=0.9, who=measles.age < 15, when=[5], name='vx'), stop=10, **kw).init()
    assert sim.interventions.vx.disease == 'measles'
    return sims


def test_timing():
    sc.heading('Testing the timing of deliveries: start= and stop= as a window, with an event in when=; and a timestep of a delivery\'s own')
    def run(solver, intv):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), init_prev=0.01, name='sir')
        pop = dict(n_agents=2000) if solver == 'abm' else dict(total_pop=2000)
        return ss.Sim(sir, interventions=intv(sir), start='2020-01-01', stop='2020-04-01', dt=ss.days(1), solver=solver, verbose=0, **pop).run()
    for solver in ['binomial', 'abm']: # On an event, within a window, as durations after the start of a sim with a calendar
        sim = run(solver, lambda sir: ss.screen(sensitivity=1.0, when=sir.infection, coverage=1.0, start=ss.days(30), stop='2020-03-01', name='test'))
        tested = np.flatnonzero(sim.results.test.delivered)
        assert tested.min() == 30 and tested.max() == 59
    a, b = [run('binomial', lambda sir: ss.modify(sir.pars.beta, multiply=0.0, **when)).results.sir.infection.values for when in [dict(start=ss.days(20), stop=ss.days(40)), dict(when=ss.schedule(start=ss.days(20), stop=ss.days(40)))]]
    assert np.array_equal(a, b) and a[20:40].sum() == 0 and a[40] > 0
    once = run('abm', lambda sir: ss.vaccinate(efficacy=1.0, when=ss.days(10), coverage=0.5, name='vx')) # One duration in when= is a time, 10 days after the start, as [ss.days(10)] is
    assert np.flatnonzero(once.results.vx.delivered).tolist() == [10]
    with pytest.raises(ValueError): # The window once
        ss.vaccinate(efficacy=0.9, start=2001, when=ss.schedule(start=2002))

    # A delivery with a timestep of its own records its results on that timeline, as agents and as compartments
    for solver,dt,index in [('abm', ss.years(2), 2), ('binomial', ss.months(6), 6)]: # On a two-year timeline, the first step after 2003 (2004); on a half-year one, 2003
        delivery = ss.vaccinate(ss.Vaccine(efficacy=0.5), when=ss.schedule(start=2003), dt=dt)
        sim = ss.Sim(n_agents=1000, total_pop=1000, diseases='sir', networks='random' if solver == 'abm' else None, solver=solver, start=2000, stop=2010, dt=ss.years(1), interventions=delivery, verbose=0).run()
        res = sim.results[delivery.name]
        assert len(res.delivered) == sim.interventions[0].t.npts != sim.t.npts
        assert res.delivered[index] > 0 and res.delivered.sum() == res.delivered[index] and res.coverage[index] == 1
    return


# %% Tests and treatment

def test_care():
    sc.heading('Testing a test whose sensitivity depends on the state and on age, and a treatment that moves people along a transition out of a state that is not infected')
    for solver,pars in [('ode', {}), ('abm', dict(n_agents=50_000))]:
        S, E, I, R, P = ss.states(5)
        seir = ss.Disease(S >> E @ 0.3*I, E >> I @ ss.days(4), I >> R @ ss.days(8), protect=ss.Transition(S, P, rate=ss.perday(0)), init=dict(I=0.01), name='seir').stratify(age=[0, 15])
        test = ss.Diagnostic(sensitivity=dict(I=ss.by(age=[0.5, 0.9]), E=0.2), specificity=1.0)
        screening = ss.screen(test, when=ss.schedule(every=10, start=20), coverage=0.5)
        shield = ss.treat(ss.Treatment(moves=seir.protect), who=seir.S, when=ss.days(0), coverage=0.5)
        res = ss.Sim(seir, interventions=[screening, shield], stop=60, solver=solver, **kw, **pars).run().results
        ti = 30 # The step from time 30
        before = lambda name, **sel: res.seir[name].select(**sel)[ti] if sel else res.seir[name][ti] # At the start of the step
        expected = 0.5*(0.5*before('I', age=0) + 0.9*before('I', age=1) + 0.2*before('E')) # Nobody else tests positive
        assert np.isclose(res.screening.positive[ti], expected, rtol=0.05 if solver == 'ode' else 0.15), solver
        assert np.isclose(res.seir.protect.sum(), 0.5*0.99*kw['total_pop'], rtol=0.05), solver
    with pytest.raises(KeyError): # Not a state of the disease
        ss.Sim(make_sis(), interventions=ss.screen(ss.Diagnostic(sensitivity=dict(X=0.5)), when=ss.schedule(every=10)), stop=20, **kw).run()
    return res


def test_coded_disease():
    sc.heading('Testing deliveries for a disease written as code: screen, treat, and vaccinate on ssl.AgentSIR')
    def make(interventions):
        sir = ssl.AgentSIR(beta=ss.peryear(0.5), init_prev=0.1, dur_inf=ss.years(5), p_death=0.0)
        return ss.Sim(n_agents=5000, diseases=sir, networks='random', interventions=interventions, start=2000, stop=2010, verbose=0).run()
    base = make([]).results.agentsir.infected
    screening = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.95), when=ss.schedule(every=ss.years(1)), coverage=0.2, name='screening')
    treatment = ss.treat(ss.Treatment(cure=0.8), when=screening.positive, name='treatment')
    vx = ss.vaccinate(ss.Vaccine(efficacy=1.0), when=[2000], coverage=0.5, name='vx') # Through rel_sus
    res = make([screening, treatment, vx]).results
    t = res.treatment
    assert t.delivered.sum() == res.screening.positive.sum() > 0 and t.unneeded.sum() > 0 # False positives are treated, and need not be
    assert np.isclose(t.cured.sum()/(t.cured.sum() + t.failed.sum()), 0.8, atol=0.05)
    assert res.vx.delivered.sum() > 0 and res.agentsir.infected.max() < 0.9*base.max()
    return res


# %% Steps of care

def make_cascade():
    """ A rapid test every week for those not tested in four weeks, a PCR to confirm (with a delay), and treatment with limited capacity """
    rapid = ss.Diagnostic(sensitivity=0.8, specificity=0.95)
    pcr = ss.Diagnostic(sensitivity=0.95, specificity=0.99, result_delay=ss.days(2))
    screening = ss.screen(rapid, who=rapid.since >= ss.days(28), when=ss.schedule(every=ss.days(7)), coverage=0.1, name='screening')
    confirm = ss.screen(pcr, when=screening.positive, coverage=0.9, name='confirm')
    treatment = ss.treat(ss.Treatment(cure=0.95), when=confirm.positive, capacity=ss.perday(50), name='treatment')
    return [screening, confirm, treatment]


def test_loop_order():
    sc.heading('Testing moving a module in the loop')
    def make():
        screen = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.98), when=ss.schedule(every=1), coverage=0.2)
        treatment = ss.treat(ss.Treatment(cure=0.95), when=screen.positive)
        return ss.Sim(make_sir(), interventions=[screen, treatment], n_agents=5000, stop=60, verbose=0, unit='day'), screen
    sim, screen = make()
    sim.run()
    labels = sim.loop.to_df().label.tolist()
    assert labels.index('screen.step') < labels.index('sir.step') # By default, interventions come before transmission

    moved, screen = make()
    moved.init()
    moved.loop.insert(screen, label='sir.step') # So someone can be infected, tested, and treated in one step
    moved.run()
    labels = moved.loop.to_df().label.tolist()
    assert labels.index('screen.step') == labels.index('sir.step') + 1 and labels.count('screen.step') == labels.count('sir.step') # Moved, not added
    assert moved.results.screen.positive.sum() > 0 and not np.array_equal(moved.results.sir.I, sim.results.sir.I)
    return moved


def test_delay():
    sc.heading('Testing delay= on a triggered step: a treatment three months after the test, fixed or drawn per person, as agents')
    def make(delay=ss.months(3), **kwargs):
        S, I = ss.states(2)
        sis = ss.Disease(infection=S >> I @ ss.peryear(2)*I, recovery=I >> S @ ss.years(1), cure=I >> S @ 0, init_prev=0.1, name='sis')
        screening = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.95), when=ss.schedule(every=ss.months(3)), coverage=0.3)
        treatment = ss.treat(ss.Treatment(cure={sis.cure: 0.9}), when=ss.on(screening.positive, delay=delay))
        return ss.Sim(sis, interventions=[screening, treatment], start=2000, stop=2005, dt=ss.months(1), total_pop=20_000, verbose=0, **kwargs)
    check_agents(make) # Who is waiting is each person's own, so agents only
    res = make(solver='abm', n_agents=20_000).run().results
    pos, tx = res.screening.positive[:-3], res.treatment.delivered[3:] # Treated three steps after the test, from the first step
    assert tx[0] > 0 and tx[1:3].sum() == 0
    assert np.allclose(tx, pos, rtol=1e-6), f'{tx[::3]} treated against {pos[::3]} positive'
    sim = make(delay=ss.uniform(ss.months(1), ss.months(5)), solver='abm', n_agents=20_000).run() # Drawn per person: 2 to 5 steps after the test
    tx = sim.results.treatment.delivered
    assert tx[:2].sum() == 0 and np.isclose(tx[2:5].sum(), 0.75*sim.results.screening.positive[0], rtol=0.1) # Those due after 2, 3, or 4 steps
    return sim


def test_delay_cycle():
    sc.heading('Testing a cycle of triggers through a delayed step: a test repeated a year after each negative result')
    def make(loop='confirm', delay=ss.years(1)):
        S, I = ss.states(2)
        sis = ss.Disease(infection=S >> I @ ss.peryear(0)*I, recovery=I >> S @ ss.years(1), init_prev=0, name='sis') # Nobody infected: every result negative
        entry = ss.screen(ss.Diagnostic(specificity=1, name='dx0'), when=[2001], name='entry')
        retest = ss.screen(ss.Diagnostic(specificity=1, name='dx1'), when=ss.on(entry.negative, delay=delay), name='retest')
        confirm = ss.screen(ss.Diagnostic(specificity=1, name='dx2'), when=retest.negative, name='confirm')
        retest._trigger = entry.negative | (confirm.negative if loop == 'confirm' else retest.negative)
        return ss.Sim(sis, interventions=[entry, confirm, retest], start=2000, stop=2005, dt=ss.months(1), n_agents=500, solver='abm', verbose=0).run()
    for loop in ['confirm', 'self']: # Through a step that follows it, and through itself
        res = make(loop).results
        assert np.array_equal(np.flatnonzero(res.entry.delivered), [12]) # The results of the step that starts at 2001
        assert np.array_equal(np.flatnonzero(res.retest.delivered), [24, 36, 48, 60]) and np.all(res.retest.delivered[[24, 36, 48, 60]] == 500), loop # The last at the stop, where the last step starts
    res = make(delay=ss.constant(0)).results # A wait drawn as zero ends on the next step
    assert np.array_equal(np.flatnonzero(res.retest.delivered)[:3], [13, 14, 15])
    a = ss.screen(ss.Diagnostic(), when=ss.schedule(every=7), name='a') # Without a delay, a cycle raises
    b = ss.screen(ss.Diagnostic(), when=a.positive, name='b')
    a._trigger = b.positive
    with pytest.raises(ValueError):
        ss.Sim(make_sir(), interventions=[a, b], **kw)
    return res


def test_save_load():
    sc.heading('Steps with memory and a delayed test, saved and loaded mid-run, end as if they had not been')
    sim = ss.Sim(make_sis(), interventions=make_cascade(), solver='abm', stop=120, **kw).init()
    sim.run(until=60)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'care.sim')
        sim.save(path)
        loaded = ss.load(path)
    loaded.run()
    sim.run()
    assert np.array_equal(sim.results.sis.I.values, loaded.results.sis.I.values)
    assert np.array_equal(sim.results.treatment.delivered.values, loaded.results.treatment.delivered.values)
    return loaded


def test_agents_only():
    sc.heading('Testing the deliveries that are agents only: an explicit ODE refuses each before the run, and solver="auto" runs it as agents')
    def make(form, **kwargs):
        S, I = ss.states('S I')
        sis, other = make_sis(), ss.Disease(infection=ss.Transition(S, I, rate=0.2 * I), init_prev=0.01, name='other')
        test = ss.Diagnostic(sensitivity=0.9, result_delay=ss.days(3) if form == 'result_delay' else None)
        scr = ss.screen(test, when=ss.schedule(every=7), coverage=0.5, disease='sis', name='scr')
        rescr = ss.screen(test, when=scr.negative, disease='sis', name='rescr') # The same test again, for some
        drug = ss.Treatment(cure=0.9)
        steps = dict(
            queue        = ss.treat(drug, when=scr.positive, capacity=ss.capacity(10, queue=True)),
            delay        = ss.treat(drug, when=ss.on(scr.positive, delay=ss.days(14))),
            result_delay = ss.treat(drug, when=scr.positive),
            since        = ss.treat(drug, who=scr.since > ss.days(30), when=scr.positive),
            count        = ss.treat(drug, who=scr.positive.n_times < 2, when=scr.positive),
            ever         = ss.treat(drug, who=~scr.negative.ever, when=scr.positive), # Kept: a level of people
            union        = ss.treat(drug, when=scr.positive | rescr.positive),
            product      = ss.treat(drug, when=test.positive), # Delivered by two steps
            waning       = ss.vaccinate(ss.Vaccine(efficacy=0.9, waning=lambda t: np.exp(-t/50)), when=[10]),
            diseases     = ss.vaccinate(ss.Vaccine(efficacy=0.9), when=[10], disease=['sis', 'other']),
        )
        return ss.Sim([sis] + [other]*(form == 'diseases'), interventions=[scr, rescr, steps[form]], stop=20, **kw, **kwargs)
    for form in ['queue', 'delay', 'result_delay', 'since', 'count', 'union', 'product', 'waning', 'diseases']:
        with pytest.raises(ss.CapabilityError):
            make(form, solver='ode').init()
        assert isinstance(make(form).init().solver, ss.solvers.abm), form
    assert isinstance(make('ever').run().solver, ss.solvers.ode)

    # A follow-up of a cure finds everyone it reached, the cured and the rest (each a part of its event), under every solver
    def follow(**kwargs):
        scr = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.5), when=[5], name='scr') # False positives are not infected: unneeded, in S with the cured
        tx = ss.treat(ss.Treatment(cure=0.5), when=scr.positive, name='tx')
        after = ss.vaccinate(ss.Vaccine(efficacy=0.5), when=tx, name='after')
        return ss.Sim(make_sis(), interventions=[scr, tx, after], **kwargs)
    for solver,sim in run_each(follow, stop=10).items():
        r = sim.results
        assert r.tx.unneeded[5] > 0 and np.isclose(r.after.delivered[5], r.tx.delivered[5], rtol=0.01 if solver == 'binomial' else 1e-6), solver # The binomial solver draws
    return sim


if __name__ == '__main__':
    sc.options(interactive=False)
    for name,test in list(globals().items()):
        if name.startswith('test_'):
            test()
