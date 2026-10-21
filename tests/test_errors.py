"""
Test that each named error, warning, and note (ss.errors.classes()) is raised, warned, or noted by the smallest model or call that should give it
"""

# %% Imports and settings
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import starsim.library as ssl
import starsim.interop

kw = dict(verbose=0)
triggers = {} # The trigger of each class, by class
notes = {} # For a note, a distinctive part of its message in sim.notes


def trigger(cls, note=None):
    """ Register the function that triggers a class; for a note, the function returns the sim, whose notes include note= """
    def register(func):
        triggers[cls] = func
        if note is not None:
            notes[cls] = note
        return func
    return register


# Classes with no trigger: none at present. Each would be listed here, with why, e.g. {ss.Example: 'raised nowhere: remove it'}
untriggered = {}


# %% Models

def sir(name='sir', beta=0.3, **kwargs):
    """ A simple SIR """
    S, I, R = ss.states('S I R')
    return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, dur=10), name=name, **sc.mergedicts(dict(init_prev=0.01), kwargs))


def ode(*args, **kwargs):
    """ A sim run as compartments (by default, as an ODE) """
    return ss.Sim(*args, **sc.mergedicts(dict(solver='ode'), kw, kwargs))


def fittable(beta=None, **kwargs):
    """ An SIR with a parameter to fit, and data from beta=0.3 (every 10 days) """
    beta = sc.ifelse(beta, ss.par(0.3, bounds=[0.1, 0.5]))
    truth = ode(sir(), unit='day', stop=60).run().results.sir.n_I
    data = pd.DataFrame(dict(n_I=truth.values[::10]), index=truth.timevec[::10])
    return ode(sir(beta=beta), unit='day', stop=60, data=data, **kwargs)


# %% Errors

@trigger(ss.IncompatibleTypes)
def _(): 1 - ss.peryear(0.1)

@trigger(ss.WrongDimension)
def _(): ss.peryear(0.1) * ss.peryear(0.1)

@trigger(ss.ReservedName)
def _():
    I, R = ss.states('I R')
    ss.Disease(step=ss.Transition(I, R, dur=3))

@trigger(ss.UnknownState)
def _():
    S, I = ss.states('S I')
    ss.Disease(S >> I @ 0.1*I, init=dict(Q=0.1))

@trigger(ss.BadBranchSplit)
def _():
    I, H, R = ss.states('I H R')
    ss.Disease(a=ss.Transition(I, H, p=0.2, dur=3), b=ss.Transition(I, R, p=0.7, dur=3))

@trigger(ss.DuplicateDeclaration)
def _(): ode(ss.SIR(name='a', births=ss.birth(rate=0.02)), ss.SIS(name='b', births=ss.birth(rate=0.02))).init()

@trigger(ss.NonlinearExposure)
def _():
    S, I = ss.states('S I')
    ss.Disease(S >> I @ 0.1*I*I)

@trigger(ss.UndrivenTransition)
def _():
    S, I = ss.states('S I')
    ss.Disease(births=ss.Transition(ss.null, S), infection=S >> I @ 0.1*I)

@trigger(ss.UnnamedState)
def _():
    states = ss.states(2)
    ss.Disease(states[0] >> states[1] @ 0.1)

@trigger(ss.ExposureToCount)
def _():
    S, I = ss.states('S I')
    W = ss.State('W', people=False)
    ss.Disease(infection=S >> I @ 0.5*W, shedding=ss.null >> W @ 10*I.n)

@trigger(ss.UnknownTransition)
def _(): sir().stratify(age=[0, 15], adjust={'recovry': [1, 2]})

@trigger(ss.BadSplit)
def _(): ode(sir().stratify(age=[0, 15], split=[0.2, 0.3, 0.5])).init()

@trigger(ss.DuplicateStructuralDimension)
def _(): sir().stratify(age=[0, 15]).stratify(age=[0, 20])

@trigger(ss.InvalidAdjustment)
def _(): ode(sir().stratify(age=[0, 15], adjust={'recovery': [1, -1]})).init()

@trigger(ss.PartialStratificationMixing)
def _(): ode(sir().stratify(age=[0, 15], states=['S', 'R']), networks=ss.contacts([[3, 1], [1, 3]], bands=[0, 15])).init()

@trigger(ss.InconsistentDimension)
def _(): ode([sir('a').stratify(age=[0, 15]), sir('b').stratify(age=[0, 20])]).init()

@trigger(ss.NoGeography)
def _(): ode(sir().stratify(patch=['a', 'b'], mobility=ss.gravity())).init()

@trigger(ss.RouteMismatch)
def _(): ss.Sim(sir(beta=ss.peract(0.01)), networks='random', n_agents=100, **kw).init()

@trigger(ss.UnmodifiableTarget)
def _(): ode(sir(), connectors=ss.modify('sir.nothing', multiply=2)).init()

@trigger(ss.UnknownEffect)
def _(): ode(sir(), interventions=ss.vaccinate(ss.Vaccine(efficacy=dict(aquisition=0.5)))).init()

@trigger(ss.UnknownDisease)
def _(): ode(sir(), interventions=ss.vaccinate(ss.Vaccine(efficacy=0.5), at=[5], disease='flu')).init()

@trigger(ss.UnknownParameter)
def _(): sir(gamma=0.1)

@trigger(ss.ShapeError)
def _(): ode(sir().stratify(age=[0, 15], adjust={'recovery': [1, 2, 3]})).init()

@trigger(ss.UnvaluedParameter)
def _(): ode(sir(beta=ss.par())).init()

@trigger(ss.UnknownLocation)
def _(): ss.people('Kenyaa')

@trigger(ss.OutOfBounds)
def _(): ss.peract(1.5)

@trigger(ss.UnknownOutcomeSource)
def _(): ode(sir(cases=ss.outcome('infections'))).init()

@trigger(ss.OutcomeCycle)
def _():
    late = ss.outcome('infection', name='late')
    early = ss.outcome(late, name='early')
    late.source = early
    ode(sir(late=late, early=early)).init()

@trigger(ss.DataMisaligned)
def _(): ode(sir(beta=ss.par(0.3, bounds=[0.1, 0.5])), data=pd.DataFrame(dict(time=[500], n_I=[10])), stop=100).calibrate()

@trigger(ss.UnknownDataColumn)
def _(): ode(sir(), data=pd.DataFrame(dict(time=[50], n_Q=[10])), stop=100).init()

@trigger(ss.OverlappingSources)
def _():
    S, I, R, D = ss.states('S I R D')
    d = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), death=ss.Transition(I, D, dur=50), init_prev=0.01, name='sird')
    ode(d, interventions=ss.treat(ss.Treatment(moves=[d.recovery, d.death]))).init()

@trigger(ss.TriggerLoop)
def _():
    S, A, B = ss.states('S A B')
    ode(ss.Disease(S, A, B, sa=S >> A @ 0.1, ab=ss.Transition(A, B, p=1.0), ba=ss.Transition(B, A, p=1.0), name='loop'), stop=5).run()

@trigger(ss.ClockOutsideDomain)
def _():
    S, A, B, C = ss.states('S A B C')
    ss.Disease(S, A, B, C, ab=A >> B @ ss.days(4), sc_=ss.Transition(S, C, rate=ss.perday(1)*(B.time > 1)))

@trigger(ss.NoScheduledDwell)
def _():
    S, A, B, C = ss.states('S A B C')
    ss.Disease(S, A, B, C, ab=A >> B @ ss.perday(0.2), bc=ss.Transition(A, C, rate=ss.perday(1)*A.progress))

@trigger(ss.PeopleAndCounts)
def _():
    S, X = ss.states('S X')
    X.people = False
    ss.Disease(S, X, births=ss.Transition(ss.null, [S, X], rate=ss.peryear(100), p=[0.9, 0.1]))

@trigger(ss.InitWhereNobodyIs)
def _():
    A, B = ss.states('A B')
    d = ss.Disease(A, B, ab=A >> B @ ss.peryear(0), init=dict(B=ss.by(np.array([[0, 0.5], [0, 0.5]]), dims=['age', 'ever'])), name='d').stratify(ever=[False, True], split=[1, 0])
    ode(d, stratify=dict(age=[0, 30]), stop=1).init()

@trigger(ss.CapabilityError)
def _(): ss.Sim(sir(beta=ss.seasonal(mean=0.15, amplitude=0.5)), solver='gillespie', approx=False, unit='day', stop=10, **kw).init()

@trigger(ss.TimeNotKept)
def _(): ss.Sim(sir(), n_agents=100, stop=5, **kw).run().diseases.sir.ti_R.notnan

@trigger(ss.TimeOutsideSim)
def _():
    d = sir()
    ode(d, interventions=ss.change(d.pars.beta, at=[2030], to=[0.1]), start='2020-01-01', stop='2020-04-30').init()

@trigger(ss.MissingDimension)
def _(): ode(sir().stratify(age=[0, 15], where=dict(sex='female'))).init()

@trigger(ss.NothingToFit)
def _(): ode(sir(), data=pd.DataFrame(dict(time=[50], n_I=[10])), stop=100).calibrate()

@trigger(ss.NoData)
def _(): ode(sir(beta=ss.par(0.3, bounds=[0.1, 0.5]))).calibrate()

@trigger(ss.NoPosterior)
def _(): fittable().calibrate(method='nuts')


# %% Warnings

@trigger(ss.TargetStatsNotRecovered)
def _(): ss.Sim(sir(), networks=ssl.DegreeNet(n_contacts=0.7, dur=1000), n_agents=200, stop=10, **kw).run().networks[0].check(n_runs=2)

@trigger(ss.AtBound)
def _(): fittable(beta=ss.par(0.1, bounds=[0.05, 0.15])).calibrate()

@trigger(ss.StageStopped)
def _(): fittable().calibrate(method=[dict(method='nelder-mead', maxiter=2, scan=0), 'nelder-mead'])

@trigger(ss.NotConverged)
def _(): fittable(solver='binomial', total_pop=10_000).calibrate(method='pmcmc', likelihood=dict(n_I=ss.poisson()), draws=10, warmup=10, chains=2, particles=10, parallel=False, rand_seed=1).summarize() # Warned when the fit is summarized

@trigger(ss.UndrivenUnused)
def _():
    S, I = ss.states('S I')
    ode(ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), cure=ss.Transition(I, S), init_prev=0.01, name='sis')).init()

@trigger(ss.LossyExport)
def _(): ss.Sim(sir(), networks='random', n_agents=100, **kw).to_code()

@trigger(ss.ImportIncomplete)
def _():
    src = ss.interop.Source('si', format='test')
    src.add_state('S', init=99); src.add_state('I', init=1); src.add_par('beta', 0.3)
    src.add_flow('S', 'I', rate='beta * I / (S + I)')
    src.skip('season', 'a time function')
    ss.interop.build(src, verbose=0)


# %% Notes

@trigger(ss.UnitInferred, note="unit='day'")
def _(): return ode(sir(), start='2020-01-01', stop='2020-03-01').init()

@trigger(ss.LargeStep, note='shorter than the step')
def _():
    S, I, R = ss.states('S I R')
    return ode(ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=0.5), init_prev=0.01, name='sir')).init()

@trigger(ss.MatrixAggregated, note='aggregated from 3 age groups')
def _(): return ode(sir(), stratify=dict(age=[0, 15]), networks=ss.contacts(np.ones((3, 3)), bands=[0, 15, 50])).init()

@trigger(ss.OverlappingSet, note='the last one wins')
def _():
    d = sir()
    return ode(d, connectors=[ss.modify(d.pars.beta, set=0.1, name='a'), ss.modify(d.pars.beta, set=0.2, name='b')], stop=5).run()

@trigger(ss.ClippedToRange, note='made it negative')
def _():
    d = sir()
    return ode(d, connectors=ss.modify(d.pars.beta, add=-1), stop=5).run()

@trigger(ss.SmallCounts, note='matter of chance')
def _(): return ode(sir(init_prev=0.002), total_pop=1000, stop=10).run()

@trigger(ss.ExtinctionPossible, note='would usually die out')
def _():
    S, I, R = ss.states('S I R')
    return ode(ss.Disease(infection=ss.Transition(S, I, rate=0.5 * I), recovery=ss.Transition(I, R, dur=5), waning=ss.Transition(R, S, dur=300), init_prev=0.01, name='sirs'), unit='day', stop=1000, total_pop=10_000).run()

@trigger(ss.StagesUnderAuto, note='approximated by')
def _(): return ode(ss.SIR(dur_inf=ss.lognorm_ex(mean=ss.years(6))), stop=2001).init()

@trigger(ss.Unidentifiable, note='only multiplied together')
def _():
    beta, c = ss.par(0.1, bounds=[0.01, 1], name='beta'), ss.par(4, bounds=[1, 10], name='c')
    S, I, R = ss.states('S I R')
    model = lambda data=None: ode(ss.Disease(S >> I @ beta*c*I, I >> R @ ss.days(8), init=dict(I=0.01), name='sir'), unit='day', stop=60, data=data)
    truth = model().run().results.sir.n_I
    return model(pd.DataFrame(dict(n_I=truth.values[::10]), index=truth.timevec[::10])).calibrate(pars=['beta', 'c']).sim


# %% Define the tests

def test_every_class():
    sc.heading('Testing that every named error, warning, and note has a trigger')
    classes = set(ss.errors.classes())
    missing = classes - set(triggers) - set(untriggered)
    assert not missing, f'No trigger in test_errors.py for {sc.strjoin(sorted(c.__name__ for c in missing))}'
    assert set(triggers) | set(untriggered) == classes, 'A trigger is for a class that is no longer in ss.errors.classes()'
    return


@pytest.mark.parametrize('cls', list(triggers), ids=lambda cls: cls.__name__)
def test_trigger(cls):
    func = triggers[cls]
    if issubclass(cls, ss.Note):
        sim = func()
        assert any(notes[cls] in note for note in sim.notes), f'Expected a note with "{notes[cls]}" for {cls.__name__}, not {sim.notes}'
    elif issubclass(cls, ss.StarsimWarning):
        with pytest.warns(cls):
            func()
    else:
        with pytest.raises(cls) as info:
            func()
        assert info.type is cls, f'Expected {cls.__name__}, not {info.type.__name__}: {info.value}'
    return


# %% Run as a script
if __name__ == '__main__':
    T = sc.timer()
    test_every_class()
    for cls in triggers:
        test_trigger(cls)
    T.toc()
