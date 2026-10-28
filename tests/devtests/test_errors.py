"""
Test that each error, warning, and note is raised, warned, or noted, with its message, by the smallest model or call that should give it
"""

# %% Imports and settings
import re
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import starsim.interop

kw = dict(verbose=0)
triggers = {} # The trigger of each message, by a distinctive part of it
kinds = {} # What each gives: an exception class, a warning class, or 'note' (a note is in sim.notes)


def trigger(match, kind):
    """ Register the function that gives a message containing `match`, of the given kind; for a note, the function returns the sim, whose notes include it """
    def register(func):
        assert match not in triggers, f'Two triggers for "{match}"'
        triggers[match] = func
        kinds[match] = kind
        return func
    return register


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
    truth = ode(sir(), unit='day', stop=60).run().results.sir.I
    data = pd.DataFrame(dict(I=truth.values[::10]), index=truth.timevec[::10])
    return ode(sir(beta=beta), unit='day', stop=60, data=data, **kwargs)


# %% Errors

@trigger('a number - a rate (ss.peryear) has no meaning', TypeError)
def _(): 1 - ss.peryear(0.1)

@trigger('is in units of per time squared', TypeError)
def _(): ss.peryear(0.1) * ss.peryear(0.1)

@trigger('since it is an attribute of every module', ValueError)
def _():
    I, R = ss.states('I R')
    ss.Disease(step=ss.Transition(I, R, dur=3))

@trigger('init= names a state "Q"', KeyError)
def _():
    S, I = ss.states('S I')
    ss.Disease(S >> I @ 0.1*I, init=dict(Q=0.1))

@trigger('the branch probabilities out of state "I" sum to 0.9, not 1', ValueError)
def _():
    I, H, R = ss.states('I H R')
    ss.Disease(a=ss.Transition(I, H, p=0.2, dur=3), b=ss.Transition(I, R, p=0.7, dur=3))

@trigger('people are born and die once', ValueError)
def _(): ode(ss.SIR(name='a', births=ss.birth(rate=0.02)), ss.SIS(name='b', births=ss.birth(rate=0.02))).init()

@trigger('is not linear in its states', ValueError)
def _():
    S, I = ss.states('S I')
    ss.Disease(S >> I @ 0.1*I*I)

@trigger('has no rate. Give it one', ValueError)
def _():
    S, I = ss.states('S I')
    ss.Disease(births=ss.Transition(ss.null, S), infection=S >> I @ 0.1*I)

@trigger('of the module has no name', ValueError)
def _():
    states = ss.states(2)
    ss.Disease(states[0] >> states[1] @ 0.1)

@trigger('a count carries no strain', ValueError)
def _():
    S, I, R = ss.states('S I R')
    d = ss.Disease(S, I, R, inf=S >> I @ ss.perday(0.3)*I.frac, rec=I >> R @ ss.days(10), strains=ss.Strains(['a', 'b']), init_prev=0.01, name='d')
    ss.Sim(d, n_agents=100, unit='day', stop=5, **kw).init()

@trigger('there is no contact with a count', ValueError)
def _():
    S, I = ss.states('S I')
    W = ss.State('W', people=False)
    ss.Disease(infection=S >> I @ 0.5*W, shedding=ss.null >> W @ 10*I.n)

@trigger('adjust= names a transition "recovry"', KeyError)
def _(): sir().stratify(age=[0, 15], adjust={'recovry': [1, 2]})

@trigger('split= has 3 entries', ValueError)
def _(): ode(sir().stratify(age=[0, 15], split=[0.2, 0.3, 0.5])).init()

@trigger('is already stratified by "age", in another way', ValueError)
def _(): sir().stratify(age=[0, 15]).stratify(age=[0, 20])

@trigger('has a negative multiplier', ValueError)
def _(): ode(sir().stratify(age=[0, 15], adjust={'recovery': [1, -1]})).init()

@trigger('so there is no level of it for the people in I to mix as', ValueError)
def _(): ode(sir().stratify(age=[0, 15], states=['S', 'R']), networks=ss.contacts([[3, 1], [1, 3]], ages=[0, 15])).init()

@trigger('A property of people has the same groups for every module', ValueError)
def _(): ode([sir('a').stratify(age=[0, 15]), sir('b').stratify(age=[0, 20])]).init()

@trigger('which has no geography', ValueError)
def _(): ode(sir().stratify(patch=['a', 'b'], mobility=ss.geo.gravity())).init()

@trigger('has no acts, so there is no act to be per', ValueError)
def _(): ss.Sim(sir(beta=ss.peract(0.01)), networks='random', n_agents=100, **kw).init()

@trigger('sir.nothing, was not found', KeyError)
def _(): ode(sir(), connectors=ss.modify('sir.nothing', multiply=2)).init()

@trigger('would not be applied by any solver', ValueError)
def _(): ode(ss.Disease(recovery=ss.Transition(*ss.states('I R'), dur=10), init=dict(I=1.0), first='I', name='d'), connectors=ss.modify('d.rel_trans', multiply=0.5)).init()

@trigger('which is not something it can act on', KeyError)
def _(): ode(sir(), interventions=ss.vaccinate(ss.Vaccine(efficacy=dict(aquisition=0.5)))).init()

@trigger('which is not a disease in the sim', KeyError)
def _(): ode(sir(), interventions=ss.vaccinate(ss.Vaccine(efficacy=0.5), at=[5], disease='flu')).init()

@trigger('it has no parameter called "gamma"', KeyError)
def _(): sir(gamma=0.1)

@trigger('has 3 values, but "age" has 2 levels', ValueError)
def _(): ode(sir().stratify(age=[0, 15], adjust={'recovery': [1, 2, 3]})).init()

@trigger('the parameter "beta" of "sir" has no value', ValueError)
def _(): ode(sir(beta=ss.par())).init()

@trigger('the place "Kenyaa" was not found', KeyError)
def _(): ss.people('Kenyaa')

@trigger('A probability per act must be between 0 and 1', ValueError)
def _(): ss.peract(1.5)

@trigger('is not a transition or outcome of "sir"', KeyError)
def _(): ode(sir(cases=ss.outcome('infections'))).init()

@trigger('depend on each other in a loop', ValueError)
def _():
    late = ss.outcome('infection', name='late')
    early = ss.outcome(late, name='early')
    late.source = early
    ode(sir(late=late, early=early)).init()

@trigger('lies entirely outside the simulated period', ValueError)
def _(): ode(sir(beta=ss.par(0.3, bounds=[0.1, 0.5])), data=pd.DataFrame(dict(time=[500], I=[10])), stop=100).fit()

@trigger('the data column "n_Q" matches no result', KeyError)
def _(): ode(sir(beta=ss.par(0.3, bounds=[0.1, 0.5])), data=pd.DataFrame(dict(time=[50], n_Q=[10])), stop=100).fit()

@trigger('each person treated can move along only one transition', ValueError)
def _():
    S, I, R, D = ss.states('S I R D')
    d = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), death=ss.Transition(I, D, dur=50), init_prev=0.01, name='sird')
    ode(d, interventions=ss.treat(ss.Treatment(moves=[d.recovery, d.death]))).init()

@trigger('moves with no wait out of the states of "loop" form a loop', ValueError)
def _():
    S, A, B = ss.states('S A B')
    ode(ss.Disease(S, A, B, sa=S >> A @ 0.1, ab=ss.Transition(A, B, p=1.0), ba=ss.Transition(B, A, p=1.0), name='loop'), stop=5).run()

@trigger('which has a value only for people in B', ValueError)
def _():
    S, A, B, C = ss.states('S A B C')
    ss.Disease(S, A, B, C, ab=A >> B @ ss.days(4), sc_=ss.Transition(S, C, rate=ss.perday(1)*(B.time > 1)))

@trigger('so no dwell time is scheduled', ValueError)
def _():
    S, A, B, C = ss.states('S A B C')
    ss.Disease(S, A, B, C, ab=A >> B @ ss.perday(0.2), bc=ss.Transition(A, C, rate=ss.perday(1)*A.progress))

@trigger('moves between people (S) and counts (X', ValueError)
def _():
    S, X = ss.states('S X')
    X.people = False
    ss.Disease(S, X, births=ss.Transition(ss.null, [S, X], rate=ss.peryear(100), p=[0.9, 0.1]))

@trigger('where the split puts nobody', ValueError)
def _():
    A, B = ss.states('A B')
    d = ss.Disease(A, B, ab=A >> B @ ss.peryear(0), init=dict(B=ss.by(np.array([[0, 0.5], [0, 0.5]]), dims=['age', 'ever'])), first='A', name='d').stratify(ever=[False, True], split=[1, 0])
    ode(d, stratify=dict(age=[0, 30]), stop=1).init()

@trigger('which approx=False does not allow', ss.CapabilityError)
def _(): ss.Sim(sir(beta=ss.seasonal(mean=0.15, amplitude=0.5)), solver='gillespie', approx=False, unit='day', stop=10, **kw).init()

@trigger('sir.ti_R is not kept', AttributeError)
def _(): ss.Sim(sir(), n_agents=100, stop=5, **kw).run().diseases.sir.ti_R.notnan

@trigger('is outside the sim', ValueError)
def _():
    d = sir()
    ode(d, interventions=ss.change(d.pars.beta, at=[2030], to=[0.1]), start='2020-01-01', stop='2020-04-30').init()

@trigger('but the model is not stratified by "sex"', KeyError)
def _(): ode(sir().stratify(age=[0, 15], where=dict(sex='female'))).init()

@trigger('fit() found no parameter with bounds', ValueError)
def _(): ode(sir(), data=pd.DataFrame(dict(time=[50], I=[10])), stop=100).fit()

@trigger('fit() needs data', ValueError)
def _(): ode(sir(beta=ss.par(0.3, bounds=[0.1, 0.5]))).fit()

@trigger('but not every data series has a likelihood', ValueError)
def _(): fittable().fit(method='nuts')

@trigger('Simulation has already been finalized', ss.AlreadyRunError)
def _(): ode(sir(), stop=5).run().finalize()

@trigger('has not been initialized', ss.errors.DistNotInitializedError)
def _(): ss.random(strict=True)(5)

@trigger('you will draw the same random numbers twice', ss.errors.DistSeedRepeatError)
def _():
    dist = ss.random().init(trace='x', sim=ss.Sim(n_agents=100, **kw).init())
    dist.jump(to=10)
    dist.jump(to=3) # Back to a state it has passed

@trigger('Logical operations are only valid on Boolean arrays', ss.errors.BooleanOperationError)
def _(): ss.Sim(n_agents=100, **kw).init().people.age & True


# %% Warnings

@trigger('against a target of 0.7', RuntimeWarning)
def _(): ss.Sim(sir(), networks=ss.DegreeNet(n_contacts=0.7, dur=1000), n_agents=200, stop=10, **kw).run().networks[0].check(n_runs=2)

@trigger('is at one of its bounds', ss.errors.AtBound)
def _(): fittable(beta=ss.par(0.1, bounds=[0.05, 0.15])).fit()

@trigger('so the stages after it were not run', RuntimeWarning)
def _(): fittable().fit(method=[dict(method='nelder-mead', maxiter=2, scan=0), 'nelder-mead'])

@trigger('the chains disagree about beta', ss.errors.NotConverged)
def _(): fittable(solver='binomial', total_pop=10_000).fit(method='pmcmc', likelihood=dict(I=ss.poisson()), draws=10, warmup=10, chains=2, particles=10, parallel=False, rand_seed=1).summarize() # Warned when the fit is summarized

@trigger('so only custom code can move people along it', RuntimeWarning)
def _():
    S, I = ss.states('S I')
    ode(ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), cure=ss.Transition(I, S), init_prev=0.01, name='sis')).init()

@trigger('not exported: network', ss.errors.LossyExport)
def _(): ss.interop.export(ss.Sim(sir(), networks='random', n_agents=100, **kw), 'odin')

@trigger('season: a time function', ss.errors.ImportIncomplete)
def _():
    src = ss.interop.Source('si', format='test')
    src.add_state('S', init=99); src.add_state('I', init=1); src.add_par('beta', 0.3)
    src.add_flow('S', 'I', rate='beta * I / (S + I)')
    src.skip('season', 'a time function')
    ss.interop.build(src, verbose=0)


# %% Notes

@trigger("unit='day'", 'note')
def _(): return ode(sir(), start='2020-01-01', stop='2020-03-01').init()

@trigger('shorter than the step', 'note')
def _():
    S, I, R = ss.states('S I R')
    return ode(ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=0.5), init_prev=0.01, name='sir')).init()

@trigger('aggregated from 3 age groups', 'note')
def _(): return ode(sir(), stratify=dict(age=[0, 15]), networks=ss.contacts(np.ones((3, 3)), ages=[0, 15, 50])).init()

@trigger('the last one wins', 'note')
def _():
    d = sir()
    return ode(d, connectors=[ss.modify(d.pars.beta, set=0.1, name='a'), ss.modify(d.pars.beta, set=0.2, name='b')], stop=5).run()

@trigger('made it negative', 'note')
def _():
    d = sir()
    return ode(d, connectors=ss.modify(d.pars.beta, add=-1), stop=5).run()

@trigger('matter of chance', 'note')
def _(): return ode(sir(init_prev=0.002), total_pop=1000, stop=10).run()

@trigger('would usually die out', 'note')
def _():
    S, I, R = ss.states('S I R')
    return ode(ss.Disease(infection=ss.Transition(S, I, rate=0.5 * I), recovery=ss.Transition(I, R, dur=5), waning=ss.Transition(R, S, dur=300), init_prev=0.01, name='sirs'), unit='day', stop=1000, total_pop=10_000).run()

@trigger('approximated by', 'note')
def _(): return ode(ss.SIR(dur_inf=ss.lognorm_ex(mean=ss.years(6))), stop=2001).init()

@trigger('only multiplied together', 'note')
def _():
    beta, c = ss.par(0.1, bounds=[0.01, 1], name='beta'), ss.par(4, bounds=[1, 10], name='c')
    S, I, R = ss.states('S I R')
    model = lambda data=None: ode(ss.Disease(S >> I @ beta*c*I, I >> R @ ss.days(8), init=dict(I=0.01), name='sir'), unit='day', stop=60, data=data)
    truth = model().run().results.sir.I
    return model(pd.DataFrame(dict(I=truth.values[::10]), index=truth.timevec[::10])).fit(pars=['beta', 'c']).sim

@trigger('runs as Python', 'note')
def _():
    @ss.jit(nopython=False)
    def uncompilable(x):
        return sc.findinds(x > 0) # Numba can't call Sciris
    with ss.options.context(verbose=1), sc.capture() as said: # With no sim, the note is printed, not kept
        uncompilable(np.arange(3))
    return sc.objdict(notes=said.splitlines()) # In place of the sim's notes


# %% Define the tests

@pytest.mark.parametrize('match', list(triggers))
def test_trigger(match):
    func, kind = triggers[match], kinds[match]
    if kind == 'note':
        sim = func()
        assert any(match in note for note in sim.notes), f'Expected a note with "{match}", not {sim.notes}'
    elif issubclass(kind, Warning):
        with pytest.warns(kind, match=re.escape(match)):
            func()
    else:
        with pytest.raises(kind, match=re.escape(match)) as info:
            func()
        if kind.__module__ == ss.errors.__name__: # One of Starsim's own: that class, not a subclass of it
            assert info.type is kind, f'Expected {kind.__name__}, not {info.type.__name__}: {info.value}'
    return


# %% Run as a script
def test_unknown_par_set():
    """ MG7: a parameter set before init() that the module never defined raises, with a did-you-mean """
    net = ss.RandomNet()
    net.pars.n_contact = 4
    with pytest.raises(KeyError, match='did you mean n_contacts'):
        ss.Sim(diseases='sis', networks=net, n_agents=100, verbose=0).init()
    net = ss.RandomNet()
    net.pars.update(extra=1, create=True) # Defined on purpose
    ss.Sim(diseases='sis', networks=net, n_agents=100, verbose=0).init()
    sis = ss.SIS()
    sis.pars.betta = 0.9
    with pytest.raises(KeyError, match='did you mean beta'):
        ss.Sim(diseases=sis, networks='random', n_agents=100, verbose=0).init()
    return


def test_rebound_par():
    """ FP13: a subclass that rebinds a declared parameter to a method or a value raises, since its transitions never read it """
    with pytest.raises(ss.CapabilityError, match='as a method'):
        class P(ss.Pregnancy):
            def loss_rate(self, sim, uids):
                return 0
    with pytest.raises(ss.CapabilityError, match='as a value'):
        class S(ss.SIR):
            beta = 0.2
    return


if __name__ == '__main__':
    T = sc.timer()
    for match in triggers:
        test_trigger(match)
    test_unknown_par_set()
    test_rebound_par()
    T.toc()
