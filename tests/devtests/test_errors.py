"""
Test that each kind of error and warning (each class), and a few notes, is raised, warned, or noted by the smallest model or call that should give it
"""

# %% Imports and settings
import re
import pytest
import pandas as pd
import sciris as sc
import starsim as ss

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

@trigger('is not linear in its states', ValueError)
def _():
    S, I = ss.states('S I')
    ss.Disease(S >> I @ 0.1*I*I)

@trigger('sir.nothing, was not found', KeyError)
def _(): ode(sir(), connectors=ss.modify('sir.nothing', multiply=2)).init()

@trigger('has not been initialized', ss.errors.DistNotInitializedError)
def _(): ss.random(strict=True)(5)

@trigger('you will draw the same random numbers twice', ss.errors.DistSeedRepeatError)
def _():
    dist = ss.random().init(trace='x', sim=ss.Sim(n_agents=100, **kw).init())
    dist.jump(to=10)
    dist.jump(to=3) # Back to a state it has passed

@trigger('has 3 edges (in p1), but 2 values of "p2"', ValueError)
def _(): ss.Sim(sir(), networks=ss.Network(edges=dict(p1=[0, 1, 2], p2=[3, 4])), n_agents=100, **kw).init()

@trigger('Logical operations are only valid on Boolean arrays', ss.errors.BooleanOperationError)
def _(): ss.Sim(n_agents=100, **kw).init().people.age & True


# %% Warnings

@trigger('against a target of 0.7', RuntimeWarning)
def _(): ss.Sim(sir(), networks=ss.DegreeNet(n_contacts=0.7, dur=1000), n_agents=200, stop=10, **kw).run().networks[0].check(n_runs=2)

@trigger('is at one of its bounds', ss.errors.AtBound)
def _(): fittable(beta=ss.par(0.1, bounds=[0.05, 0.15])).fit()

@trigger('the chains disagree about beta', ss.errors.NotConverged)
def _(): fittable(solver='binomial', total_pop=10_000).fit(method=ss.fitting.method('pmcmc', draws=10, warmup=10, chains=2, particles=10, parallel=False), likelihood=dict(I=ss.poisson()), rand_seed=1).summarize() # Warned when the fit is summarized

@trigger('not exported: network', ss.errors.LossyExport)
def _(): ss.export_to(ss.Sim(sir(), networks='random', n_agents=100, **kw), 'odin')

@trigger('season: a time function', ss.errors.ImportIncomplete)
def _():
    src = ss.converters.Source('si', format='test')
    src.add_state('S', init=99); src.add_state('I', init=1); src.add_par('beta', 0.3)
    src.add_flow('S', 'I', rate='beta * I / (S + I)')
    src.skip('season', 'a time function')
    ss.converters.build(src, verbose=0)


# %% Notes

@trigger("unit='day'", 'note')
def _(): return ode(sir(), start='2020-01-01', stop='2020-03-01').init()

@trigger('matter of chance', 'note')
def _(): return ode(sir(init_prev=0.002), total_pop=1000, stop=10).run()


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


def test_unknown_pars():
    """ A parameter set before init() that the module never defined raises, unless defined on purpose; a subclass that rebinds a declared parameter raises, since its transitions never read it """
    net = ss.RandomNet()
    net.pars.n_contact = 4
    with pytest.raises(KeyError, match='did you mean n_contacts'):
        ss.Sim(diseases='sis', networks=net, n_agents=100, verbose=0).init()
    net = ss.RandomNet()
    net.pars.update(extra=1, create=True)
    ss.Sim(diseases='sis', networks=net, n_agents=100, verbose=0).init()
    with pytest.raises(ss.CapabilityError, match='as a method'):
        class P(ss.Pregnancy):
            def loss_rate(self, sim, uids):
                return 0
    return


if __name__ == '__main__':
    T = sc.timer()
    for match in triggers:
        test_trigger(match)
    test_unknown_pars()
    T.toc()
