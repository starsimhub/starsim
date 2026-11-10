"""
Test interventions on declared models: modifiers, conditions, products, and their delivery
"""

# %% Imports and settings
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import starsim.library as ssl

kw = dict(verbose=0, unit='day', total_pop=50_000)
solvers = ['ode', 'binomial', 'abm']


def make_sir(**kwargs):
    S, I, R = ss.states('S I R')
    sir = ss.Disease(
        infection = ss.Transition(S, I, rate=0.3 * I),
        recovery  = ss.Transition(I, R, dur=10),
        init_prev = 0.01,
        name = 'sir',
        **kwargs,
    )
    return sir


def make_sis():
    S, I = ss.states('S I')
    sis = ss.Disease(
        infection = ss.Transition(S, I, rate=0.3 * I),
        recovery  = ss.Transition(I, S, dur=10),
        init_prev = 0.01,
        name = 'sis',
    )
    return sis


def run_each(make, stop=150, **kwargs):
    """ Run the sim that `make` returns with each solver """
    return sc.obj({solver:make(solver=solver, stop=stop, **kw, **kwargs).run() for solver in solvers})


def check_close(sims, func, rtol=0.1):
    """ Check that a number from each sim matches the ODE's """
    vals = sc.obj({solver:func(sim) for solver,sim in sims.items()})
    assert np.allclose(vals[:], vals.ode, rtol=rtol), f'Expected the solvers to agree, not {vals}'
    return vals


def make_cascade(**kwargs):
    """ A rapid test every week for those not tested in four weeks, a PCR to confirm (with a delay), and treatment with limited capacity """
    rapid = ss.Diagnostic(sensitivity=0.8, specificity=0.95)
    pcr = ss.Diagnostic(sensitivity=0.95, specificity=0.99, result_delay=ss.days(2))
    screening = ss.screen(rapid, who=rapid.since >= ss.days(28), when=ss.schedule(every=ss.days(7)), coverage=0.1, name='screening') # Named, since a subclass of ss.Sim may make them
    confirm = ss.screen(pcr, when=screening.positive, coverage=0.9, name='confirm')
    treatment = ss.treat(ss.Treatment(cure=0.95), when=confirm.positive, capacity=ss.perday(50), name='treatment')
    return [screening, confirm, treatment]


def check_agents(make, **kwargs):
    """ Check that a model is agents only: an explicit ODE refuses it before the run, and solver='auto' runs it as agents """
    with pytest.raises(ss.CapabilityError):
        make(solver='ode', **kwargs).init()
    assert isinstance(make(**kwargs).init().solver, ss.solvers.abm)
    return


# %% Define the tests

def test_change():
    sc.heading('Testing changes at given times: ss.modify() of a time series')
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

    # Values instead of multipliers, and dates
    sir = make_sir()
    to = ss.modify(sir.pars.beta, set=ss.timeseries({'2020-01-21': 0.09, '2020-03-01': 0.3}), when=ss.schedule(start='2020-01-21'))
    dated = ss.Sim(sir, interventions=to, start='2020-01-01', stop='2020-04-30', total_pop=50_000, verbose=0).run()
    assert np.allclose(dated.results.sir.S, res.S)
    sir = make_sir()
    start = ss.date('2020-01-01')
    after = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, set=ss.timeseries({start + ss.days(20): 0.09, start + ss.days(60): 0.3}), when=ss.schedule(start=start + ss.days(20))), start=start, stop='2020-04-30', total_pop=50_000, verbose=0).run()
    assert np.allclose(after.results.sir.S, res.S) # A time after the start is the start plus a duration
    after = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, set=ss.timeseries({ss.days(20): 0.09, ss.days(60): 0.3}), start=ss.days(20)), start=start, stop='2020-04-30', total_pop=50_000, verbose=0).run()
    assert np.allclose(after.results.sir.S, res.S) # So is a duration alone, that long after the start
    with pytest.raises(ValueError, match='calendar year'): # In a sim with dates, a bare number is a year: 20 is the year 20, outside the sim
        ss.Sim(sir, interventions=ss.modify(sir.pars.beta, set=0.09, when=ss.schedule(start=20)), start='2020-01-01', stop='2020-04-30', verbose=0).init()
    late = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, set=0.09, when=ss.schedule(start=2030)), start='2020-01-01', stop='2020-04-30', verbose=0).init() # A start after the end never applies, so a sim can be run to an earlier stop
    assert any('never applies' in note for note in late.notes)
    ss.Sim(sir, interventions=ss.modify(sir.pars.beta, set=0.09, when=ss.schedule(start='2020-04-30')), start='2020-01-01', stop='2020-04-30', verbose=0).init() # At the stop is fine, since the last step starts there
    late = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, set=0.09, when=ss.schedule(start='2020-05-01')), start='2020-01-01', stop='2020-04-30', verbose=0).init() # After the stop, where no step starts, so it never applies
    assert any('never applies' in note for note in late.notes)

    # It is a modifier, so a window and a function of the date work
    for when,dates in [(ss.schedule(start=0, stop=0), {}), (ss.schedule(lambda now: False, start='2000-01-02'), dict(start='2000-01-01'))]: # A function of the date needs a calendar
        sir = make_sir()
        sim = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.3, when=when), solver='ode', dur=ss.days(120), **dates, **kw).run()
        assert np.isclose(sim.results.sir.infection.cum[-1], base.results.sir.infection.cum[-1]) # Not 492 infections against 9,309

    # A modifier applies whatever form the parameter takes
    forms = dict(scalar=0.3, func=lambda sim: 0.3, timeseries=ss.timeseries({0: 0.3, 100: 0.3}), seasonal=ss.seasonal(mean=0.3, amplitude=0.2, peak=0))
    for solver in ['ode', 'abm']:
        for key,beta in forms.items():
            S, I, R = ss.states('S I R')
            sir = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
            stopped = ss.modify(sir.pars.beta, multiply=0.0)
            sim = ss.Sim(sir, interventions=stopped, solver=solver, stop=50, **kw).run()
            assert sim.results.sir.infection.sum() == 0, f'{key} under {solver}'

    # On a beta for each route, with a pair of directions on one (it crashed)
    S, I = ss.states('S I')
    d = ss.Disease(infection=ss.Transition(S, I, rate=dict(random=ss.peryear(0.3), sexualnet=ss.peract(m2f=0.1, f2m=0.2)) * I), init_prev=0.05, name='d')
    ss.Sim(d, networks=['random', ss.SexualNet()], interventions=ss.modify('d.pars.beta', multiply=0.5, when=ss.schedule(start=2001)), n_agents=1000, start=2000, stop=2003, dt=ss.years(0.25), verbose=0).run()
    return sims


def test_applied():
    sc.heading('Testing that every modifier and product is applied, or refused when the sim is made')
    # Refused: an efficacy on a transition with several destinations
    P, C, R = ss.states('P C R')
    hpv = ss.Disease(precin=ss.Transition(P, {C: 0.2, R: 0.8}, dur=ss.years(2)), init=dict(P=1.0), default_state='P', name='hpv')
    with pytest.raises(ValueError, match="'precin.C'"):
        ss.Sim(hpv, interventions=ss.vaccinate(ss.Vaccine(efficacy=dict(precin=0.5)), when=0), **kw).init()

    # Applied: modifiers of a disease written as code with its own transmit(), and a change to an inflow under the ODE
    class OwnTransmit(ssl.AgentSIS):
        def transmit(self): # One pass for each of two strains, each with half the infectiousness, as Covasim's is by variant
            return super().transmit(rel_trans=0.5*self.transmissibility()) | super().transmit(rel_trans=0.5*self.transmissibility())
    sim = ss.Sim(diseases=OwnTransmit(name='sis'), networks='random', connectors=ss.modify('sis.rel_sus', multiply=0), n_agents=1000, verbose=0).run()
    assert sim.results.sis.infection.sum() == 0
    for solver in ['ode', 'abm']:
        Sv = ss.State(people=False)
        mosq = ss.Disease(ss.State('S'), Sv, emergence=ss.unborn >> Sv @ ss.perday(100), mortality=Sv >> ss.dead @ ss.days(10), init=dict(Sv=1000), name='mosq')
        sim = ss.Sim(mosq, interventions=ss.modify(mosq.emergence, multiply=0.5, when=ss.schedule(start=10)), solver=solver, n_agents=1000, total_pop=1000, unit='day', stop=100, verbose=0).run()
        assert np.isclose(sim.results.mosq.Sv[-1], 500, rtol=0.05), solver
    return sim


def test_conditions():
    sc.heading('Testing conditions')
    ppl = ss.People(100)
    sir = make_sir().stratify(age=[0, 15, 65])
    cond = (ppl.age < 15) & sir.S
    assert isinstance(cond, ss.conditions.Cond)
    assert str(cond) == '((people.age < 15) & sir.S)'
    assert str((ss.person.age < 15) & sir.S) == str(cond) # The same condition with no people made
    assert str(ppl.age + 1 > 3) == str(ss.person.age + 1 > 3) and str(ppl.age.isin([1, 2])) == str(ss.person.age.isin([1, 2])) # Arithmetic and isin() are lazy too
    assert not hasattr(ss.People, 'age') # ss.People is the class only: the lazy references are ss.person
    with pytest.raises(TypeError):
        bool(cond)

    # The same condition selects agents or groups
    def make(eligible=cond, **kwargs):
        vaccine = ss.Vaccine(efficacy=0.9)
        kids = ss.vaccinate(vaccine, who=eligible, when=[5])
        return ss.Sim(sir, interventions=kids, **kwargs)
    sims = run_each(make, stop=20)
    for sim in sims.values():
        n_S = sim.results.sir.S.by('age').values[5, 0]
        assert np.isclose(sim.results.kids.delivered[5], n_S, rtol=0.02) # Every susceptible child at time 5, and nobody else, in the step from it
        assert np.allclose(sim.results.kids.delivered.by('age').values[5], [sim.results.kids.delivered[5], 0, 0]) # By stratum: all in the youngest group
        total = sim.results.kids.delivered.by('age').summarize() # Labeled by group
        assert isinstance(total, pd.Series) and len(total) == 3 and np.isclose(total.iloc[0], sim.results.kids.delivered.summarize())
    check_close(sims, lambda sim: sim.results.kids.delivered.sum(), rtol=0.05)

    # A function of the sim does the same
    func = make(eligible=lambda sim: sim.people.age < 15, solver='ode', stop=20, **kw).run()
    assert func.results.kids.delivered[5] > sims.ode.results.kids.delivered[5] # Including the children who are not susceptible

    # An age that is not the edge of a group cannot be used on groups, so the model runs as agents unless told otherwise
    auto = make(eligible=ppl.age < 10, stop=20, **kw).init()
    assert isinstance(auto.solver, ss.solvers.abm) and 'falls inside a bin' in auto.solver_reason
    with pytest.raises(ss.CapabilityError):
        make(eligible=ppl.age < 10, solver='ode', stop=20, **kw).run()

    # The model's own age groups, compared with an age, are compared with their edges
    for solver in ['ode', 'abm']:
        delivered = [make(eligible=e & sir.S, solver=solver, stop=20, **kw).run().results.kids.delivered.sum() for e in [sir.age < 15, sir.age == '0-14']]
        assert np.isclose(delivered[0], delivered[1]) and np.isclose(delivered[0], sims[solver].results.kids.delivered.sum()) # Not everyone
    with pytest.raises(ValueError):
        make(eligible=sir.age < 10, solver='abm', stop=20, **kw).run()
    return sims


def test_transfer():
    sc.heading('Testing vaccination as a transfer along a stratification')
    def make(**kwargs):
        sir = make_sir().stratify(vaccinated=[False, True]) # A boolean dimension starts with everyone False
        rollout = ss.Event(dict(vaccinated=True), who=~sir.vaccinated, when=[5, 30], p=0.7)
        return ss.Sim(sir, interventions=rollout, connectors=ss.modify(sir.infection, multiply=ss.by(vaccinated=[1.0, 0.15])), **kwargs)
    sims = run_each(make)
    for sim in sims.values():
        delivered = sim.results.rollout.delivered
        assert np.allclose(delivered[[5, 30]], [35_000, 10_500], rtol=0.05) # 70% of everyone, then 70% of those left (not of the half that started unvaccinated)
    check_close(sims, lambda sim: sim.results.sir.infection.cum[-1], rtol=0.35) # What is left after the campaign depends on how far the epidemic had got, which varies by about 10% between agent-based runs, and agents are about 10% below the ODE at a step of a day
    with pytest.raises(TypeError): # ss.transfer() is movement at a rate, never a delivery
        ss.transfer(dict(vaccinated=True), when=[5], coverage=0.7)

    # From one level to another, given as an arrow
    sir = make_sir().stratify(risk={'low': 1, 'high': 0})
    sim = ss.Sim(sir, interventions=ss.Event(dict(risk=('low', 'high')), when=[5], p=0.4, name='riskier'), solver='ode', stop=10, **kw).run()
    assert np.isclose(sim.results.riskier.delivered[5], 0.4*kw['total_pop'], rtol=0.01)
    return sims


def test_deliver():
    sc.heading('Testing delivery by a function, triggers, and scenarios')

    # A function that says who is eligible, as any condition in who= does: UIDs, as agents
    def give(sim):
        return sim.people.auids[:10_000]
    def by_function(**kwargs):
        sir = make_sir()
        vaccine = ss.Vaccine(efficacy=0.9)
        chosen = ss.vaccinate(vaccine, who=give, when=5, capacity=8000)
        return ss.Sim(sir, interventions=chosen, **kwargs)
    sim = by_function(solver='abm', stop=20, **kw).run()
    assert np.isclose(sim.results.chosen.delivered.sum(), 8000) # Capped by the capacity

    # A trigger: vaccinate while there are many infections
    def triggered(**kwargs):
        sir = make_sir()
        vaccine = ss.Vaccine(efficacy=0.9)
        many = lambda sim: sim.results.sir.I[sim.ti] > 2000 # At the start of the step
        response = ss.vaccinate(vaccine, who=many, when=ss.schedule(every=1), coverage=0.2)
        return ss.Sim(sir, interventions=response, **kwargs)
    sims = run_each(triggered)
    for sim in sims.values():
        res = sim.results
        given = np.flatnonzero(res.response.delivered)
        assert len(given) and np.all(res.sir.I[given] > 2000)

    # Rounds a year apart on weekly steps fall a year apart, each on the first step of the year (they were 52 weeks apart, so 2005 had two)
    yearly = ss.vaccinate(ss.Vaccine(efficacy=0.9), when=ss.schedule(every=ss.years(1)), coverage=0.2, name='yearly')
    sim = ss.Sim(make_sir(), interventions=yearly, start='2000-01-01', stop='2010-01-01', dt=ss.weeks(1), solver='ode', verbose=0).init()
    starts = [sim.t.tvec[ti] for ti in sorted(sim.interventions.yearly.round_tis)]
    assert [d.year for d in starts] == list(range(2000, 2010)) and all(d.month == 1 and d.day <= 7 for d in starts)

    # Scenarios are sims, and can be compared run by run
    def scenario(vaccinate=True, **kwargs):
        sir = make_sir()
        campaign = ss.vaccinate(ss.Vaccine(efficacy=0.9), when=[5], coverage=0.6)
        return ss.Sim(sir, interventions=campaign if vaccinate else None, solver='binomial', stop=100, **kw)
    msim = ss.parallel({'baseline': scenario(False), 'vaccine': scenario(True)}, n_runs=3, workers=1)
    diff = msim.diff('baseline').vaccine.sir.infection.cum
    assert diff.values.shape == (3, 101)
    assert np.all(diff.values[:, -1] < 0) # Fewer infections in every run
    fig = msim.plot('sir.I', show_interventions=False) # The median and an interval for each scenario
    assert len(fig.axes[0].lines) == 2
    return msim


def test_coinfection():
    sc.heading('Testing two diseases that modify each other')
    def make(coinfect=True):
        S, I = ss.states('S I')
        hiv = ss.Disease(infection=ss.Transition(S, I, rate=dict(sexual=ss.peract(m2f=0.002, f2m=0.001)) * I), init_prev=0.05)
        S, I = ss.states('S I')
        syph = ss.Disease(infection=ss.Transition(S, I, rate=0.01 * I), cure=ss.Transition(I, S, dur=ss.years(3)), init_prev=0.05)
        coinfection = [
            ss.modify(hiv.rel_trans, multiply=20.0, who=syph.I, name='hiv_trans'), # Both diseases have a state called I, so each condition remembers its own
            ss.modify(syph.rel_sus, multiply=5.0, who=hiv.I, name='syph_sus'),
        ]
        return ss.Sim(diseases=[hiv, syph], networks=ss.SexualNet(), connectors=coinfection if coinfect else None, start=2000, stop=2020, verbose=0)
    sim = make().run()
    base = make(coinfect=False).run()
    assert list(sim.diseases.keys()) == ['hiv', 'syph'] # Named from their variables
    assert sim.results.hiv.I[-1] > base.results.hiv.I[-1]
    assert sim.results.syph.infection.cum[-1] > base.results.syph.infection.cum[-1] # About 11% more on average over 10 seeds, but from 2% (seed 1) to 21% between single runs
    assert 'who=syph.I' in sim.what_modifies('hiv.rel_trans')
    assert 'who=hiv.I' in sim.what_modifies('syph.rel_sus')
    return sim


def test_loop_order():
    sc.heading('Testing moving a module in the loop')
    def make():
        test = ss.Diagnostic(sensitivity=0.9, specificity=0.98)
        drug = ss.Treatment(cure=0.95)
        screen = ss.screen(test, when=ss.schedule(every=1), coverage=0.2)
        treatment = ss.treat(drug, when=screen.positive)
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
    assert not np.array_equal(moved.results.sir.I, sim.results.sir.I)
    return moved


def test_triggers():
    sc.heading('Testing interventions triggered by the state of the sim')
    pars = dict(stop=200, **kw)
    def make():
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.001, name='sir')

    # A function of the sim has no memory, so it switches on and off as the number infected crosses the threshold
    sir = make()
    distancing = ss.modify(sir.pars.beta, multiply=0.3, who=lambda sim: sim.results.sir.I[sim.ti] > 1000)
    bare = ss.Sim(sir, interventions=distancing, **pars).run()
    base = ss.Sim(make(), **pars).run()
    assert isinstance(bare.solver, ss.solvers.ode)
    assert bare.results.sir.I.max() < 0.2*base.results.sir.I.max()

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
            if not self.closed and sim.results.sir.I[sim.ti] > self.threshold: # During a step, sim.ti is its own index, and its stocks (the state at its start) are recorded already
                self.closed = True
                self.reopen = sim.ti + 7*self.weeks
            self.days_closed += self.closed
            return

    sir = make()
    effect = ss.modify(sir.pars.beta, multiply=0.3, who=lambda sim: sim.interventions.closure.closed)
    held = ss.Sim(sir, interventions=[Closure(name='closure'), effect], **pars).run()
    assert isinstance(held.solver, ss.solvers.ode)
    assert held.interventions.closure.days_closed % 28 == 0 or held.interventions.closure.closed # Closed for four weeks at a time
    assert held.results.sir.I.max() < base.results.sir.I.max()
    assert np.isclose(sir.pars.beta.value, 0.3) # The base value is never overwritten

    # Or until= a duration: a time that holds on one step keeps the modifier on for ten, as a window would
    sir = make()
    once = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.3, when=ss.schedule([50]), until=10), **pars).run() # The step that starts at 50
    sir = make()
    timed = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.3, when=ss.schedule(start=50, stop=60)), **pars).run()
    assert np.allclose(once.results.sir.I, timed.results.sir.I)
    return held


def test_delivery_inputs():
    sc.heading('Testing that a delivery says what is wrong with its product and coverage')
    with pytest.raises(ValueError, match='between 0 and 1'):
        ss.vaccinate(efficacy=0.9, coverage=1.5)
    with pytest.raises(ValueError, match='between the states'): # Not a product, so read as a move by its paths
        ss.vaccinate('vaccine')
    return


@sc.timer()
def test_coverage_by_group():
    sc.heading('Testing that the coverage of several groups is what was delivered over who was eligible in them')
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.2 * I), recovery=ss.Transition(I, R, dur=10), init_n=dict(I=10), name='sir')
    sir.stratify(age=[0, 5, 15])
    vx = ss.vaccinate(ss.Vaccine(efficacy=0.9), when=[5], coverage=ss.by(age=[0.8, 0.2, 0.5]), name='vx')
    res = ss.Sim(sir, interventions=vx, unit='day', stop=20, solver='ode', verbose=0).run().results.vx
    young = ['0-4', '5-14']
    assert np.allclose(res.coverage.by('age').values[5], [0.8, 0.2, 0.5])
    assert np.isclose(res.coverage.select(age=young)[5], res.delivered.select(age=young)[5]/res.eligible.select(age=young)[5]) # Not 0.8 + 0.2
    assert np.isclose(res.eligible.summarize(), res.eligible.values.sum()) # The offers made over the run (it was their mean over every step, e.g. 19.6 of 1,000 eligible)
    return res


def test_sensitivity_by_state():
    sc.heading('Testing a test whose sensitivity depends on the state, and on age within a state')
    def make(**kwargs):
        S, E, I, R = ss.states(4)
        seir = ss.Disease(S >> E @ 0.3*I, E >> I @ ss.days(4), I >> R @ ss.days(8), init=dict(I=0.01), name='seir').stratify(age=[0, 15])
        test = ss.Diagnostic(sensitivity=dict(I=ss.by(age=[0.5, 0.9]), E=0.2), specificity=1.0)
        screening = ss.screen(test, when=ss.schedule(every=10, start=20), coverage=0.5)
        return ss.Sim(seir, interventions=screening, stop=60, **sc.mergedicts(dict(total_pop=50_000), kwargs))
    for solver,pars in [('ode', {}), ('abm', dict(n_agents=50_000))]:
        res = make(solver=solver, **kw, **pars).run().results
        ti = 30 # The step from time 30
        before = lambda name, **sel: res.seir[name].select(**sel)[ti] if sel else res.seir[name][ti] # At the start of the step
        expected = 0.5*(0.5*before('I', age=0) + 0.9*before('I', age=1) + 0.2*before('E')) # Nobody else tests positive
        print(f'{solver}: {res.screening.positive[ti]:.0f} positive, expected {expected:.0f}')
        assert np.isclose(res.screening.positive[ti], expected, rtol=0.05 if solver == 'ode' else 0.15)
    with pytest.raises(KeyError, match='which is not a state of "sis"'):
        ss.Sim(make_sis(), interventions=ss.screen(ss.Diagnostic(sensitivity=dict(X=0.5)), when=ss.schedule(every=10)), stop=20, **kw).run()
    return res


def test_treat_any_transition():
    sc.heading('Testing a treatment that moves people along a transition out of a state that is not infected')
    for solver,pars in [('ode', {}), ('abm', dict(n_agents=10_000))]:
        S, I, R, P = ss.states(4)
        sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), protect=ss.Transition(S, P, rate=ss.perday(0)), init=dict(I=0.01), name='sir')
        shield = ss.treat(ss.Treatment(moves=sir.protect), who=sir.S, when=ss.days(0), coverage=0.5)
        res = ss.Sim(sir, interventions=shield, stop=100, solver=solver, **kw, **pars).run().results
        print(f'{solver}: {res.sir.protect.sum():.0f} protected')
        assert np.isclose(res.sir.protect.sum(), 0.5*0.99*kw['total_pop'], rtol=0.05)
    return res


def test_current_value():
    sc.heading('Testing base and effective values')
    sir = make_sir()
    protect = ss.modify(sir.rel_sus, multiply=0.4, who=sir.R, name='protect')
    sim = ss.Sim(sir, interventions=protect, solver='abm', n_agents=2000, stop=40, **kw).run()
    arr = sim.diseases.sir.rel_sus
    R = sim.diseases.sir.R.uids
    assert np.allclose(sim.diseases.sir.current_value('rel_sus', R), 0.4*arr[R]) # The base is untouched by the modifier
    assert 'protect' in sim.what_modifies(arr)
    ode = ss.Sim(make_sir(), interventions=ss.vaccinate(ss.Vaccine(efficacy=0.9), when=[5]), solver='ode', stop=40, **kw).run()
    with pytest.raises(ss.CapabilityError): # There are no agents to have values
        ode.diseases.sir.current_value('rel_sus')
    assert 'multiply=0.1 ' in ode.what_modifies('sir.rel_sus') # Not 0.09999999999999998
    hiv = ssl.HIV() # A disease written as code: its rel_sus is tracked, and its parameters are said not to be (it was a KeyError)
    coded = ss.Sim(hiv, networks=ss.RandomNet(), connectors=ss.modify(hiv.rel_sus, multiply=0.5, name='half'), n_agents=200, verbose=0).init()
    assert 'half' in coded.what_modifies(coded.diseases.hiv.rel_sus)
    assert 'written as code' in coded.what_modifies('hiv.pars.beta')
    return sim


def test_delay():
    sc.heading('Testing delay= on a triggered step: a treatment three months after the test, as agents')
    def make(delay=ss.months(3), **kwargs):
        S, I = ss.states(2)
        sis = ss.Disease(infection=S >> I @ ss.peryear(2)*I, recovery=I >> S @ ss.years(1), cure=I >> S @ 0, init_prev=0.1, name='sis')
        screening = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.95), when=ss.schedule(every=ss.months(3)), coverage=0.3)
        treatment = ss.treat(ss.Treatment(cure={sis.cure: 0.9}), when=ss.on(screening.positive, delay=delay))
        return ss.Sim(sis, interventions=[screening, treatment], start=2000, stop=2005, dt=ss.months(1), total_pop=20_000, verbose=0, **kwargs)
    check_agents(make) # Who is waiting is each person's own, so agents only, fixed or drawn
    check_agents(make, delay=ss.uniform(ss.months(1), ss.months(5)))
    res = make(solver='abm', n_agents=20_000).run().results
    pos, tx = res.screening.positive[:-3], res.treatment.delivered[3:] # Treated three steps after the test, from the first step
    assert tx[0] > 0 and tx[1:3].sum() == 0
    assert np.allclose(tx, pos, rtol=1e-6), f'{tx[::3]} treated against {pos[::3]} positive'
    sim = make(delay=ss.uniform(ss.months(1), ss.months(5)), solver='abm', n_agents=20_000).run() # Drawn per person: 2 to 5 steps after the test
    tx = sim.results.treatment.delivered
    assert tx[:2].sum() == 0 and np.isclose(tx[2:5].sum(), 0.75*sim.results.screening.positive[0], rtol=0.1) # Those due after 2, 3, or 4 steps
    assert 'after 3 months' in make(solver='abm', n_agents=1000).init().interventions.treatment.describe()
    with pytest.raises(TypeError, match='ss.on'): # The old keyword says what replaces it
        ss.treat(ss.Treatment(cure=0.9), delay=ss.months(3))
    return sim


def test_delay_cycle():
    sc.heading('Testing a cycle of triggers through a delayed step: a test repeated a year after each negative result')
    def make(loop='confirm', delay=ss.years(1), order=None):
        S, I = ss.states(2)
        sis = ss.Disease(infection=S >> I @ ss.peryear(0)*I, recovery=I >> S @ ss.years(1), init_prev=0, name='sis') # Nobody infected: every result negative
        entry = ss.screen(ss.Diagnostic(specificity=1, name='dx0'), when=[2001], name='entry')
        retest = ss.screen(ss.Diagnostic(specificity=1, name='dx1'), when=ss.on(entry.negative, delay=delay), name='retest')
        confirm = ss.screen(ss.Diagnostic(specificity=1, name='dx2'), when=retest.negative, name='confirm')
        retest.trigger = entry.negative | (confirm.negative if loop == 'confirm' else retest.negative)
        mods = order(entry, retest, confirm) if order else [entry, confirm, retest]
        return ss.Sim(sis, interventions=mods, start=2000, stop=2005, dt=ss.months(1), n_agents=500, solver='abm', verbose=0).run()
    for loop in ['confirm', 'self']: # Through a step that follows it, and through itself
        res = make(loop).results
        assert np.array_equal(np.flatnonzero(res.entry.delivered), [12]) # The results of the step that starts at 2001
        assert np.array_equal(np.flatnonzero(res.retest.delivered), [24, 36, 48, 60]) and np.all(res.retest.delivered[[24, 36, 48, 60]] == 500), loop # The last at the stop, where the last step starts
    res = make(order=lambda e,r,c: [c, r, e]).results # Run in another order, those triggered after the delayed step has run wait just as long
    assert np.array_equal(np.flatnonzero(res.retest.delivered), [24, 36, 48, 60])
    res = make(delay=ss.constant(0)).results # A wait drawn as zero ends on the next step
    assert np.array_equal(np.flatnonzero(res.retest.delivered)[:3], [13, 14, 15])
    a = ss.screen(ss.Diagnostic(), when=ss.schedule(every=7), name='a') # Without a delay, a cycle still raises
    b = ss.screen(ss.Diagnostic(), when=a.positive, name='b')
    a.trigger = b.positive
    with pytest.raises(ValueError, match='cycle'):
        ss.Sim(make_sir(), interventions=[a, b], **kw)
    return res


def test_coded_disease():
    sc.heading('Testing deliveries for a disease written as code: screen, treat, and vaccinate on ssl.AgentSIR')
    def make(interventions, **kwargs):
        sir = ssl.AgentSIR(beta=ss.peryear(0.5), init_prev=0.1, dur_inf=ss.years(5), p_death=0.0, **kwargs)
        return ss.Sim(n_agents=5000, diseases=sir, networks='random', interventions=interventions, start=2000, stop=2010, verbose=0).run()
    base = make([]).results.agentsir.infected
    screening = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.95), when=ss.schedule(every=ss.years(1)), coverage=0.2, name='screening')
    treatment = ss.treat(ss.Treatment(cure=0.8), when=screening.positive, name='treatment')
    res = make([screening, treatment]).results
    t = res.treatment
    assert t.delivered.sum() == res.screening.positive.sum() > 0 and t.unneeded.sum() > 0 # False positives are treated, and need not be
    assert np.isclose(t.cured.sum()/(t.cured.sum() + t.failed.sum()), 0.8, atol=0.05)
    assert res.agentsir.infected.max() < 0.9*base.max()
    vx = ss.vaccinate(ss.Vaccine(efficacy=1.0), when=[2000], coverage=0.5)
    assert make(vx).results.agentsir.infected.max() < 0.9*base.max() # Through rel_sus

    with pytest.raises(KeyError, match='has no "clear_infection"'): # A part the product uses that the disease lacks
        class NoCure(ssl.AgentSIR):
            clear_infection = property(lambda self: (_ for _ in ()).throw(AttributeError('clear_infection')))
        ss.Sim(n_agents=500, diseases=NoCure(), interventions=ss.treat(ss.Treatment(cure=0.8), when=ss.schedule(every=1), coverage=0.1), verbose=0).init()
    return res


def test_save_load():
    """ Steps with memory and a delayed test, saved and loaded mid-run, end as if they had not been """
    sc.heading('Testing save and load of steps of care mid-run')
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


def test_copied_steps():
    """ Deliveries deep-copied together still trigger each other: a condition refers to the original delivery, so the order of steps finds it by name """
    sc.heading('Testing steps after a deep copy of the interventions')
    class CopySim(ss.Sim):
        def __init__(self, interventions=None, **kwargs):
            super().__init__(**kwargs)
            self.pars.update(dict(interventions=sc.dcp(interventions)))
    def make(sim_class):
        S, I = ss.states('S I')
        d = ss.Disease(infection=S >> I @ ss.permonth(0.3)*I, recovery=I >> S @ ss.months(6), init_prev=0.1, name='d')
        visit = ss.Event(when=ss.ref('d.I').enter, name='visit')
        test = ss.screen(ss.Diagnostic(sensitivity=0.9), when=visit, name='test')
        tx = ss.treat(ss.Treatment(cure=0.9), when=test.positive, name='tx')
        return sim_class(diseases=d, interventions=[visit, test, tx], n_agents=1000, start=2000, stop=2002, dt=ss.months(1), verbose=0).run()
    plain, copied = make(ss.Sim), make(CopySim)
    assert copied.results.tx.delivered.sum() > 0
    assert np.array_equal(plain.results.tx.delivered.values, copied.results.tx.delivered.values)
    return copied


def test_agents_only():
    sc.heading('Testing the deliveries that are agents only: an explicit ODE refuses each before the run, naming it, and solver="auto" runs it as agents')
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
        with pytest.raises(ss.CapabilityError, match='Run with solver="abm"'):
            make(form, solver='ode').init()
        assert isinstance(make(form).init().solver, ss.solvers.abm), form
    assert isinstance(make('ever').run().solver, ss.solvers.ode)

    # A follow-up of a cure finds everyone it reached, the cured and the rest (each a part of its event), under every solver
    def follow(**kwargs):
        sis = make_sis()
        scr = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.5), when=[5], name='scr') # False positives are not infected: unneeded, in S with the cured
        tx = ss.treat(ss.Treatment(cure=0.5), when=scr.positive, name='tx')
        after = ss.vaccinate(ss.Vaccine(efficacy=0.5), when=tx, name='after')
        return ss.Sim(sis, interventions=[scr, tx, after], **kwargs)
    for solver,sim in run_each(follow, stop=10).items():
        r = sim.results
        assert r.tx.unneeded[5] > 0 and np.isclose(r.after.delivered[5], r.tx.delivered[5], rtol=0.01 if solver == 'binomial' else 1e-6), solver # Was 11% short as an ODE, the parts taken as independent; the binomial solver draws
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
    for sim in sims.values(): # At birth: the share of the newborns (as agents, it was none)
        assert np.isclose(sim.results.bcg.delivered.sum(), 0.6*sim.results.sir.births.sum(), rtol=0.1)

    # at_age= never switches to agents: it needs an age group that starts at the age
    with pytest.raises(ss.CapabilityError, match='needs people.s ages'):
        make('auto', strata=None).init()
    with pytest.raises(ss.CapabilityError):
        make('auto', strata=[0, 1, 5]).init()
    assert isinstance(make('abm', strata=None, n_agents=1000).init().solver, ss.solvers.abm)

    # With two diseases, a condition on one's own age groups says which disease a delivery is for
    def sir(name):
        S, I, R = ss.states('S I R')
        return ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init_prev=0.01, name=name).stratify(age=[0, 15, 100])
    measles, flu = sir('measles'), sir('flu')
    sim = ss.Sim(diseases=[measles, flu], interventions=ss.vaccinate(efficacy=0.9, who=measles.age < 15, when=[5], name='vx'), stop=10, **kw).init()
    assert sim.interventions.vx.disease == 'measles'
    return sims



def test_own_timestep():
    """ A delivery with a timestep of its own records its results on that timeline, as agents and as compartments """
    for solver in ['abm', 'binomial']:
        for dt,index in [(ss.years(2), 2), (ss.months(6), 6)]: # The step that starts at 2003 or, on a two-year timeline, the first after it (2004)
            delivery = ss.vaccinate(ss.Vaccine(efficacy=0.5), when=ss.schedule(start=2003), dt=dt)
            sim = ss.Sim(n_agents=1000, total_pop=1000, diseases='sir', networks='random' if solver == 'abm' else None, solver=solver, start=2000, stop=2010, dt=ss.years(1), interventions=delivery, verbose=0).run()
            res = sim.results[delivery.name]
            assert len(res.delivered) == sim.interventions[0].t.npts != sim.t.npts
            assert res.delivered[index] > 0 and res.delivered.sum() == res.delivered[index]
            assert res.coverage[index] == 1
    return


def test_modify_until():
    """ until= keeps a modifier on from where= until a lower level, under each kind of solver """
    def run(solver, pop=1e6, until=True):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), init_prev=1e-3, name='sir')
        kw = dict(until=sir.I.frac <= 0.05) if until else {}
        measure = ss.modify(sir.pars.beta, multiply=0.6, who=sir.I.frac >= 0.1, **kw)
        return ss.Sim(sir, interventions=measure, total_pop=pop, dur=ss.days(200), solver=solver, verbose=0).run().results.sir.infection.values.sum()
    assert run('ode') < run('ode', until=False) # It stays on longer than the bare threshold
    assert run('abm', pop=20_000) < 20_000
    with pytest.raises(ValueError):
        ss.modify('sir.pars.beta', multiply=0.5, until=lambda sim: True)



@sc.timer()
def test_paths():
    sc.heading('Testing states by their paths in who=, where=, and move=, the same as the states, and the note on seeding beside the default')
    def run(make, solver='ode', key='interventions', **kw):
        S, E, I, R = ss.states('S E I R')
        d = ss.Disease(infection=S >> E @ ss.perday(0.4)*I, onset=E >> I @ ss.days(3), recovery=I >> R @ ss.days(7), name='seir', **kw)
        return ss.Sim(d, **{key: make(d)}, total_pop=5000, dur=ss.days(60), verbose=0, solver=solver).run()
    R = lambda sim: sim.results.seir.R[-1]
    for solver in ['ode', 'abm']:
        kw = dict(solver=solver, init_n=dict(I=10))
        assert R(run(lambda d: ss.treat(cure=1.0, who=d.I, coverage=ss.perday(0.1)), **kw)) == R(run(lambda d: ss.treat(cure=1.0, who='seir.I', coverage=ss.perday(0.1)), **kw))
        moves = [R(run(lambda d: ss.Event(move(d), when=ss.days(10), p=0.3), **kw)) for move in [lambda d: d.S >> d.R, lambda d: 'seir.S >> seir.R', lambda d: 'S >> R']]
        assert moves[0] == moves[1] == moves[2]
        mods = [R(run(lambda d: ss.modify('seir.infection', multiply=0.5, who=where(d)), key='connectors', **kw)) for where in [lambda d: d.I, lambda d: 'seir.I']]
        assert mods[0] == mods[1]
    with pytest.raises(ValueError):
        ss.Event('seir.S >> sir.R', when=[1]) # The states of two modules
    with pytest.raises(ValueError):
        ss.treat(cure=1.0, who='I') # A path needs its module
    seeding = lambda d: ss.Event(d.S >> d.I, when=ss.days(10), p=0.001)
    note = lambda sim: any('give the disease init=dict(I=0)' in note for note in sim.notes)
    sim = run(seeding)
    assert note(sim) and sim.results.seir.E[0] > 0 # The default seeding (into E, where infection puts people), with a note
    sim = run(seeding, init=dict(I=0))
    assert not note(sim) and sim.results.seir.E[0] == sim.results.seir.I[0] == 0
    S, E, I, R = ss.states('S E I R') # A network of agents runs as agents
    d = ss.Disease(infection=S >> E @ ss.peryear(10)*I, onset=E >> I @ ss.days(3), recovery=I >> R @ ss.days(7), init_n=dict(I=10), name='seir')
    assert isinstance(ss.Sim(d, networks=ssl.HybridNet(household_size=ss.poisson(mean=3)), total_pop=2000, dur=ss.days(5), verbose=0).init().solver, ss.solvers.abm)
    return

@sc.timer()
def test_start_stop():
    sc.heading('start= and stop= of a delivery or a modifier: a window, the same as in ss.schedule(), that also goes with an event in when=')
    def run(solver, intv, **pars):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), init_prev=0.01, name='sir')
        pop = dict(n_agents=2000) if solver == 'abm' else dict(total_pop=2000)
        return ss.Sim(sir, interventions=intv(sir), start='2020-01-01', stop='2020-04-01', dt=ss.days(1), solver=solver, verbose=0, **pop, **pars).run()
    for solver in ['binomial', 'abm']: # On an event, within a window (typhoidsim's ss.screen(test, on=onset, start=2001)), as durations after the start of a sim with a calendar
        sim = run(solver, lambda sir: ss.screen(sensitivity=1.0, when=sir.infection, coverage=1.0, start=ss.days(30), stop='2020-03-01', name='test'))
        tested = np.flatnonzero(sim.results.test.delivered)
        assert tested.min() == 30 and tested.max() == 59
    a, b = [run('binomial', lambda sir: ss.modify(sir.pars.beta, multiply=0.0, **when)).results.sir.infection.values for when in [dict(start=ss.days(20), stop=ss.days(40)), dict(when=ss.schedule(start=ss.days(20), stop=ss.days(40)))]]
    assert np.array_equal(a, b) and a[20:40].sum() == 0 and a[40] > 0
    once = run('abm', lambda sir: ss.vaccinate(efficacy=1.0, when=ss.days(10), coverage=0.5, name='vx')) # One duration in when= is a time, 10 days after the start, as [ss.days(10)] is
    assert np.flatnonzero(once.results.vx.delivered).tolist() == [10]
    with pytest.raises(ValueError, match='window of its own'): # The window once
        ss.vaccinate(efficacy=0.9, start=2001, when=ss.schedule(start=2002))
    with pytest.raises(ValueError, match='window of its own'):
        ss.modify('sir.pars.beta', multiply=0.5, stop=2001, when=ss.schedule(start=2000))
    with pytest.raises(TypeError, match='start='): # A function of the time alone in who= says when: the window is start= and stop=
        ss.screen(sensitivity=0.9, who=lambda sim: sim.t.now() > 2001)
    return


if __name__ == '__main__':
    sc.options(interactive=False)
    change = test_change()
    applied = test_applied()
    conds  = test_conditions()
    trans  = test_transfer()
    sens   = test_sensitivity_by_state()
    msim   = test_deliver()
    coinf  = test_coinfection()
    loop   = test_loop_order()
    held   = test_triggers()
    cover  = test_coverage_by_group()
    anytr  = test_treat_any_transition()
    current = test_current_value()
    delay  = test_delay()
    cycle  = test_delay_cycle()
    coded  = test_coded_disease()
    agents = test_agents_only()
    saved  = test_save_load()
    copied = test_copied_steps()
    aged   = test_age_and_birth()
    paths  = test_paths()
