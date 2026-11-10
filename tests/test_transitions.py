"""
Test models written as declared states and transitions, and the solvers that run them
"""

# %% Imports and settings
import sys
import time
import _thread
import functools
import itertools
from unittest import mock
import pytest
import numpy as np
import pandas as pd
import scipy.stats as stats
import sciris as sc
import starsim as ss
import starsim.fit
import starsim.library as ssl

kw = dict(verbose=0)


def same(*sims):
    """ Whether sims give the same results: two spellings of one model """
    return ss.debugtools.check_sims_match(*[sim.run() for sim in sims])


def make_sir(name='sir', **kwargs):
    """ A simple SIR """
    S, I, R = ss.states('S I R')
    sir = ss.Disease(
        infection = ss.Transition(S, I, rate=0.3 * I),
        recovery  = ss.Transition(I, R, dur=10),
        init_prev = 0.01,
        name = name,
        **kwargs,
    )
    return sir


def make_cholera():
    """ An SIR with a water reservoir, W, which is a count and not people """
    beta = 0.5 # Infection hazard at saturating exposure
    K    = 1e5 # Concentration giving half the maximum hazard
    xi   = 10  # Shedding per infectious person per day
    S, I, R = ss.states(3)
    W = ss.State(people=False) # A count
    cholera = ss.Disease(
        infection = S >> I @ beta * W.n / (W.n + K),
        recovery  = I >> R @ ss.days(5),
        shedding  = ss.outside >> W @ xi * I.n,
        decay     = W >> ss.dead @ ss.days(30),
        init      = dict(I=0.001),
    )
    return cholera


@sc.timer()
def test_forms():
    sc.heading('Testing that every way of writing a model gives the same model')
    forms = sc.obj()

    # Operators
    S, I, R = ss.states(3)
    infection = S >> I @ 0.05 * I
    recovery = ss.Transition(I, R, dur=10)
    forms.operator = ss.Disease(infection, recovery, name='sir')

    # Fully explicit
    S, I, R = ss.states(3)
    forms.explicit = ss.Disease(
        ss.Transition(S, I, rate=0.05 * I, name='infection'),
        ss.Transition(I, R, dur=10, name='recovery'),
        name = 'sir',
    )

    # A class
    class SIR(ss.Disease):
        S, I, R = ss.states('S I R')
        infection = S >> I @ 0.05 * I
        recovery = ss.Transition(I, R, dur=10)
    forms.cls = SIR()

    # The whole flow, as most other frameworks write it
    S, I, R = ss.states(3)
    N = S + I + R
    infection = S >> I @ 0.05 * S * I / N
    recovery = ss.Transition(I, R, dur=10)
    forms.flow = ss.Disease(infection, recovery, name='sir')

    sims = sc.obj()
    for key,sir in forms.items():
        sims[key] = ss.Sim(sir, stop=50, **kw)
        assert list(sir.states.keys()) == ['S', 'I', 'R']
        assert {key:p.value for key,p in sir.pars.items()} == dict(beta=0.05, dur_I=10, init_prev=0.01) # init_prev by default
    assert same(*sims.values()), 'Expected every form to give the results of the operator form'

    # Several sources as a sum of states, in the explicit form
    S, I, R, D = ss.states(4)
    death = ss.Transition(I + R, D, rate=0.01, name='death')
    assert death.source == [I, R]
    many = ss.Disease(S >> I @ 0.05*I, I >> R @ 10, death, name='sir')
    assert [s.name for s in many.graph.transitions.death.source] == ['I', 'R']

    # A distribution times a state after "@" is a rate, as in the keyword form; a distribution of durations is still a duration
    S, I = ss.states(2)
    assert repr((S >> I @ ss.gamma(shape=0.5, scale=3.8)*I).rate) == repr(ss.Transition(S, I, rate=ss.gamma(shape=0.5, scale=3.8)*I).rate)
    with pytest.raises(TypeError):
        S >> I @ ss.lognormal(mean=ss.days(8), std=ss.days(2))*I

    # Density-dependent: the whole flow without the total is the count, and either way 1% start infected by default
    density = []
    for flow in [True, False]:
        S, I, R = ss.states(3)
        density.append(ss.Sim(ss.Disease(S >> I @ (3e-5*S*I if flow else 3e-5*I.n), I >> R @ 10, name='sir'), stop=50, **kw))
    assert same(*density)
    assert density[0].results.sir.I[0] == 100 # init_prev=0.01 by default
    graph = density[0].diseases.sir.graph
    assert 'infection' in graph.transitions and 'beta' in graph.pars # A transmission like any other: it was named S_to_I, with an unnamed coefficient
    assert 'from I (density-dependent)' in repr(graph.transitions)

    # The total, with include_sum=True: beta*I/N is literally the share infectious, beta*I.frac (frequency-dependent), as is the whole flow beta*S*I/N; well mixed, the same as contact, beta*I
    freq = []
    for rate in [lambda S,I,N: 0.3*I, lambda S,I,N: 0.3*I.frac, lambda S,I,N: 0.3*I/N, lambda S,I,N: 0.3*S*I/N, lambda S,I,N: 0.3*I.n/(S.n + I.n + R.n)]:
        S, I, R, N = ss.states('S I R', include_sum=True)
        assert repr(N) == 'S + I + R'
        freq.append(ss.Sim(ss.Disease(S >> I @ rate(S, I, N), I >> R @ 10, name='sir'), stop=50, **kw))
    assert same(*freq)
    for sim in freq:
        graph = sim.diseases.sir.graph
        assert 'infection' in graph.transitions and 'beta' in graph.pars and graph.init_prev is not None
        assert [s.name for s in graph.infected] == ['I']
    assert all(repr(sim.diseases.sir.graph.transitions.infection.rate) == 'beta * I.frac' for sim in freq[1:4]) # Written with counts, it is kept as written
    assert 'from I (frequency-dependent)' in repr(freq[1].diseases.sir.graph.transitions)

    # On a network, only contact can be used: the share or number of everyone would ignore whom people meet, so is refused
    for rate in [lambda S,I,N: 0.3*I.frac, lambda S,I,N: 0.3*S*I/N, lambda S,I,N: 3e-4*I.n]:
        S, I, R, N = ss.states('S I R', include_sum=True)
        with pytest.raises(ValueError, match='write the bare state'):
            ss.Sim(ss.Disease(S >> I @ rate(S, I, N), I >> R @ 10, name='sir'), networks=ss.RandomNet(), n_agents=100, stop=5, verbose=0).init()
    return forms


@sc.timer()
def test_names():
    sc.heading('Testing how states, transitions, parameters, and modules get their names')
    beta = ss.par(0.3, bounds=[0.1, 0.5])
    S, I, R = ss.states(3)
    infection = S >> I @ beta * I
    recovery = I >> R @ ss.days(10)
    sir = ss.Disease(infection, recovery, I >> S @ ss.days(100))
    assert list(sir.graph.transitions.keys()) == ['infection', 'recovery', 'I_to_S'] # The last has no variable, so gets a fallback name
    assert sorted(sir.pars.keys()) == ['beta', 'dur_I_to_S', 'dur_recovery', 'init_prev'] # Two durations out of I, so each is qualified by its transition; init_prev by default
    assert sir.infection is infection
    A, B = ss.states(2)
    first = second = A >> B @ 0.3 * B
    assert list(ss.Disease(first).graph.transitions.keys()) == ['infection'] # Which name is meant is not known, so the only transmission is the infection
    S, E, I, R = ss.states('S E I R')
    seir = ss.Disease(S >> E @ 0.3 * I, E >> I @ ss.days(5), I >> R @ ss.days(10), I >> S @ 0.01)
    assert list(seir.graph.transitions.keys()) == ['infection', 'E_to_I', 'I_to_R', 'I_to_S'] # The rest are built from their endpoints
    assert sorted(seir.pars.keys()) == ['beta', 'dur_E', 'dur_I', 'init_prev', 'rate_I_to_S'] # A dwell time is named by its state, and a rate by its transition
    S, I, H, R, D = ss.states('S I H R D')
    death = H >> D @ ss.perday(0.02)
    discharge = H >> R @ ss.perday(0.1)
    covid = ss.Disease(S >> I @ 0.3*I, I >> H @ ss.days(8), discharge, death)
    assert sorted(covid.pars.keys()) == ['beta', 'dur_I', 'init_prev', 'rate_death', 'rate_discharge'] # Each rate named by its transition, and the duration by its state

    sim = ss.Sim(sir, stop=10, **kw).init()
    assert 'sir' in sim.diseases # The module is named from its variable
    assert sim.unit == 'day' # From the typed durations

    # An ss.par written in a slot is named from the slot, as a plain value is
    S, I, R = ss.states('S I R')
    inline = ss.Disease(infection=ss.Transition(S, I, rate=ss.par(0.3, bounds=[0.1, 1]) * I), recovery=ss.Transition(I, R, dur=10))
    assert inline.pars.beta.bounds == [0.1, 1]

    # Transitions as a list, positionally or as transitions=, and a sum of flows as a keyword, which is an outcome
    S, I, R = ss.states('S I R')
    listed = ss.Disease(transitions=[ss.Transition(S, I, rate=0.3 * I, name='infection'), ss.Transition(I, R, dur=10, name='recovery')], init=dict(I=0.01))
    assert list(listed.transitions.keys()) == ['infection', 'recovery']
    S, I, R = ss.states('S I R')
    positional = ss.Disease([ss.Transition(S, I, rate=0.3 * I, name='infection'), ss.Transition(I, R, dur=10, name='recovery')], init=dict(I=0.01))
    assert list(positional.transitions.keys()) == ['infection', 'recovery']
    S, I, R = ss.states(3)
    recovery, death = I >> R @ ss.days(10), I >> ss.dead @ ss.days(100)
    dis = ss.Disease(infection=S >> I @ 0.3 * I, recovery=recovery, death=death, leaving=recovery + death, init=dict(I=0.01), name='dis')
    res = ss.Sim(dis, stop=50, **kw).run().results.dis
    assert np.allclose(res.leaving, res.recovery + res.death)

    # A set of states is their sum in arithmetic, and a sum of states can be the sources of a transition
    S, Ia, Is, Ih, R = ss.states(5)
    clinical = ss.Disease(S, Ia, Is, Ih, R, infection=S >> Ia @ 0.3*(Ia + Is + Ih), worse=Ia >> Is @ ss.perday(0.2), admitted=Is >> Ih @ ss.perday(0.1),
                          recovery=(Ia + Is + Ih) >> R @ ss.perday(0.05), share=Ih/(Ia + Is + Ih), init_prev=0.01, name='clinical')
    assert [s.name for s in clinical.recovery.source] == ['Ia', 'Is', 'Ih']
    res = ss.Sim(clinical, unit='day', stop=50, **kw).run().results.clinical
    assert np.allclose(res.share[1:], (res.Ih/(res.Ia + res.Is + res.Ih))[1:])

    # One parameter shared by models built more than once, as for scenarios: registered with each, and set by name in each sim
    R0 = ss.par(2.0, bounds=[1, 5])
    def shared(name):
        S, I, R = ss.states(3)
        return ss.Disease(infection=S >> I @ R0 * 0.1 * I, recovery=I >> R @ ss.days(10), init=dict(I=0.01), name=name)
    first, second = shared('a'), shared('b')
    assert first.graph.pars.R0 is second.graph.pars.R0
    peak = lambda sim: sim.run().diseases[0].results.I.max()
    assert peak(ss.Sim(second, stop=60, pars=dict(R0=3.0), **kw)) > 1.5*peak(ss.Sim(second, stop=60, **kw))
    assert peak(ss.Sim([shared('c'), shared('d')], stop=60, pars=dict(R0=3.0), **kw)) == peak(ss.Sim(shared('e'), stop=60, pars=dict(R0=3.0), **kw))

    # The same value used in several transitions without ss.par (here a seasonal one, as for a leaky vaccine) is one parameter too
    S, I, R, V = ss.states('S I R V')
    beta = ss.seasonal(mean=0.3, amplitude=0.2)
    leaky = ss.Disease(infection=S >> I @ beta*I, breakthrough=V >> I @ 0.3*beta*I, recovery=I >> R @ ss.days(10), init=dict(I=0.01, V=0.5))
    assert any(leaf is leaky.pars.beta for leaf in leaky.graph.transitions.breakthrough.rate.leaves), 'ss.modify(leaky.pars.beta) should change both'

    # A state with no name is an error
    with pytest.raises(ValueError):
        states = ss.states(2)
        ss.Disease(states[0] >> states[1] @ 0.1)

    # A state given by its name, not as a state, is an error
    with pytest.raises(TypeError):
        S, I = ss.states(2)
        ss.Disease(ss.Transition(S, I, rate=0.1 * I), ss.Transition(I, 'Q', dur=3))

    # Reserved names, undriven transitions, and bad splits
    with pytest.raises(ValueError, match='"step" cannot be used as the name'):
        I, R = ss.states('I R')
        ss.Disease(step=ss.Transition(I, R, dur=3))
    with pytest.raises(ValueError, match='"describe" cannot be used as the name'): # A method of ss.Disease, not of every module
        I, R = ss.states('I R')
        ss.Disease(describe=ss.Transition(I, R, dur=3))
    with pytest.raises(ValueError, match='has no rate. Give it one'): # An inflow needs a rate; a transition between states with none is undriven
        ss.Disease(recovery=ss.Transition(I, R, dur=3), births=ss.Transition(ss.unborn, S))
    with pytest.raises(ValueError, match='the branch probabilities out of state "I" sum to 0.9, not 1'):
        I, H, R = ss.states('I H R')
        ss.Disease(a=ss.Transition(I, H, p=0.2, dur=3), b=ss.Transition(I, R, p=0.7, dur=3))
    with pytest.raises(TypeError): # Several destinations are a list, not a weighted sum
        S, I, R = ss.states('S I R')
        I >> (0.9*R + 0.1*S) @ 0.1
    with pytest.raises(ValueError, match='ss.stay'): # Below 1, an error that suggests ss.stay
        I, H, R = ss.states('I H R')
        ss.Disease(a=ss.Transition(I, H, p=0.2, dur=3), b=ss.Transition(I, R, p=0.7, dur=3))
    with pytest.raises(ValueError):
        S, I = ss.states(2)
        ss.Disease(infection = S >> I @ 0.1 * I * I)

    # The named errors that no other test raises (with these, every error class is raised by some test)
    S, E, I, R = ss.states('S E I R')
    seir = ss.Disease(infection=ss.Transition(S, E, rate=ss.par(0.3, bounds=[0.1, 0.5]) * I), latency=ss.Transition(E, I, dur=5), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='seir')
    with pytest.raises(KeyError):
        ss.Sim(make_sir(), interventions=ss.vaccinate(ss.Vaccine(efficacy=0.5), when=[5], disease='flu'), **kw).init()
    cyclic = make_sir()
    late = ss.outcome(cyclic.infection, name='late')
    early = ss.outcome(late, name='early')
    late.source = early
    with pytest.raises(ValueError):
        ss.Sim(make_sir(late=late, early=early), **kw).init()
    with pytest.raises(ValueError): # Entirely after the run
        ss.Sim(seir, data=pd.DataFrame({'time': [500], 'I': [10]}), stop=100, **kw).fit()
    with pytest.raises(ValueError, match=r'fit\(\) found no parameter with bounds'):
        ss.Sim(make_sir(), data=pd.DataFrame({'time': [50], 'I': [10]}), stop=100, **kw).fit()

    # Operators that would silently drop something
    W = ss.State(people=False) # The water first: the first state of people is still S
    S, I, R = ss.states('S I R')
    water = ss.Disease(S >> I @ ss.perday(0.5)*W.n/(W.n + 1e5), I >> R @ ss.days(5), ss.unborn >> W @ 10*I.n, W >> ss.dead @ ss.days(30), init=dict(I=0.001), name='water')
    assert [s.name for s in water.graph.people_states] == ['S', 'I', 'R']
    assert make_sir(beta=0.5).pars.beta.value == 0.5 # A keyword number sets the parameter of that name
    def sir(**kwargs):
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), name='sir', **kwargs)
    seeded = [sir(**kwargs) for kwargs in [dict(init=dict(I=1.0)), dict(init=dict(I=0.01)), dict(init_prev=0.01), dict(init_n=dict(I=1))]]
    assert ss.Sim(seeded[0], total_pop=1000, **kw).init().diseases.sir.engine.y[1].sum() == 1000 # init= is a share: 1.0 is everyone
    assert ss.Sim(seeded[3], total_pop=1000, **kw).init().diseases.sir.engine.y[1].sum() == 1 # init_n= is a number of people
    with pytest.warns(RuntimeWarning, match='init_n'):
        ss.Sim(sir(init=dict(I=1)), total_pop=1000, **kw).init() # A whole number is still a share, with a warning
    with pytest.raises(ValueError, match='init_n'):
        ss.Sim(sir(init=dict(I=10)), total_pop=1000, **kw).init()
    assert same(ss.Sim(seeded[1], **kw), ss.Sim(seeded[2], **kw)) # Two spellings of one model
    S, I, R = ss.states('S I R')
    given = ss.Sim(ss.Disease(S >> I @ ss.perday(0.5)*I, I >> R @ ss.days(7), init=dict(S=0.1, I=1e-4), name='sir'), total_pop=10_000, **kw).init().diseases.sir.engine.y
    assert np.isclose(given[0].sum(), 1000) # A given S is kept (K&R 2.2 starts 10% susceptible) ...
    assert np.isclose(given[2].sum(), 8999) # ... and the rest start in R, the only state not given
    S, E, I, R = ss.states('S E I R')
    with pytest.raises(ValueError): # With E and R both open, the rest have no obvious home
        ss.Sim(ss.Disease(S >> E @ ss.perday(0.5)*I, E >> I @ ss.days(3), I >> R @ ss.days(7), init=dict(S=0.1, I=1e-4), name='seir'), **kw).init()
    with pytest.raises(KeyError, match='it has no parameter called "gamma"'): # ... and is an error if there is none
        make_sir(gamma=0.1)
    with pytest.raises(KeyError, match='init='): # A seeding keyword points to init=
        make_sir(init_infected=100)
    S, I, R = ss.states('S I R')
    with pytest.raises(TypeError): # A second driver
        I >> R @ ss.days(10) @ 0.1
    with pytest.raises(TypeError): # Arithmetic on a duration
        I >> R @ ss.days(10) * 2
    with pytest.raises(ValueError): # Levels that do not match
        ss.by(age=[1, 2, 3]) + ss.by(age=[10, 20])
    with pytest.raises(ValueError): # Stages split a dwell time
        ss.Transition(I, R, rate=0.1, stages=3)
    S, I, H, R = ss.states('S I H R')
    branches = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), severe=ss.Transition(I, H, p=0.1, dur=5), mild=ss.Transition(I, R, dur=10), name='b')
    assert 'p = 0.9' in branches.describe() # The branch that takes the rest says so
    base = S >> I
    fast, slow = base @ 0.2, base @ 0.1
    assert fast is not slow and ss.expressions.format_expr(fast.rate) == '0.2' # A partial transition can be used twice
    two = ss.Sim(make_sir(name='a'), make_sir(name='b'), **kw).init()
    assert list(two.diseases.keys()) == ['a', 'b'] # Both models, not the second as the label
    S, I, R = ss.states(3)
    infection = S >> I @ 0.3*I
    cases = ss.outcome(0.2*infection, likelihood=ss.nbinom(n=ss.par(10, bounds=[1, 100])))
    assert 'cases_n' in ss.Disease(infection, I >> R @ ss.days(10), cases, name='sir').graph.pars # A parameter inside a likelihood is named for its outcome
    assert float(ss.par(0.2)) == 0.2
    rate3 = ss.par(0.1282) # A module-level parameter in the class form keeps its name (it was "Could not find a name for the parameter")
    class Model(ss.Disease):
        S = ss.State(); I = ss.State(); H = ss.State()
        infection = S >> I @ ss.perday(0.478)*I
        course = ss.Transition(I, {H: 0.28, S: 0.72}, rate=rate3)
    assert 'rate3' in Model(init=dict(I=0.01), name='m').graph.pars

    # Transitions reused in a second disease are copied with their parameters, so the two are independent
    S, E, I, R = ss.states('S E I R')
    inf, prog, rec = S >> E @ 1.9*I, E >> I @ ss.days(10), I >> R @ ss.days(8)
    first, second = ss.Disease(inf, prog, rec), ss.Disease(inf, prog, rec, beta=1.0, name='second') # An SEIR (I read before it is filled) can be built twice
    assert first.pars.beta.value == 1.9 and second.pars.beta.value == 1.0
    ss.Sim(second, connectors=ss.modify(second.pars.beta, multiply=0.5), n_agents=200, stop=5, verbose=0).init() # The modifier finds the second's beta
    S, I, R = ss.states('S I R')
    seeded = ss.Disease(S >> I @ 1.9*I, I >> R @ ss.days(8), init=dict(I=0.01), name='seeded')
    again = ss.Disease(S >> I @ 1.9*I, I >> R @ ss.days(8), init_prev=0.01, name='again') # The first's init= stays with it (it raised "given both init_prev= and init=")
    assert seeded.I.init == 0.01 and again.I.init is None
    S, I, R = ss.states('S I R')
    dur = ss.par(ss.days(6), bounds=[ss.days(3), ss.days(10)])
    named = ss.Disease(S >> I @ 0.3*I, I >> R @ dur, R >> S @ ss.days(100), name='named') # A parameter named like a slot, beside a bare value in that slot (it was DuplicateDeclaration)
    assert named.pars.dur is dur and 'dur_R' in named.pars

    # A model can be built with no Python frame above it, as when called from R
    S, I, R = ss.states('S I R')
    dis = without_frame(ss.Disease, S >> I @ 0.3*I, I >> R @ ss.days(10), init_prev=0.01)
    sim = without_frame(ss.Sim, dis, stop=20, verbose=0)
    assert sim.run().results.sir.I.max() > 0
    return sir


def without_frame(func, *args, **kwargs):
    """ Call a function from a new thread, so no Python frame is above it, as when it is called from R """
    out, errors = [], []
    calls = itertools.starmap(functools.partial(func, *args, **kwargs), [()]) # Only C functions between the thread and func
    orig, sys.unraisablehook = sys.unraisablehook, lambda u: errors.append(u.exc_value)
    try:
        _thread.start_new_thread(out.extend, (calls,))
        for i in range(600):
            if out or errors:
                break
            time.sleep(0.05)
    finally:
        sys.unraisablehook = orig
    if errors:
        raise errors[0]
    return out[0]


@sc.timer()
def test_time():
    sc.heading('Testing how the sim gets its unit')
    sim = ss.Sim(make_sir(), stop=100, **kw).init()
    assert sim.unit is None # Nothing has a unit, so time is in timesteps
    assert sim.t.npts == 101
    tenths = ss.Sim(make_sir(), stop=100, dt=0.1, **kw)
    assert 'beta = 0.3 per unit' in tenths.describe() and 'every 0.1 units' in tenths.describe() # Not "per step", since a step is a tenth of a unit
    with pytest.raises(ValueError): # Steps have no dates (they were days underneath)
        tenths.t.datevec
    with pytest.raises(ValueError):
        S, I, R = ss.states('S I R')
        ss.Sim(ss.Disease(infection=ss.Transition(S, I, rate=ss.seasonal(0.3, 0.2) * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01), stop=100, **kw).init()
    assert ss.Sim(make_sir(), stop=100, unit='day', **kw).init().unit == 'day'
    assert ss.Sim(make_sir(), stop=10, dt=ss.weeks(1), **kw).init().unit == 'week'
    assert ss.Sim(make_sir(), start='2020-01-01', stop='2020-03-01', **kw).init().unit == 'day'
    assert ss.Sim(make_sir(), start=2020, stop=2030, **kw).init().unit == 'year'
    assert ss.Sim(make_sir(), unit='day', dur=ss.days(3650), **kw).init().t.npts == 3651 # A duration is the length
    yearlike = ss.Sim(make_sir(), unit='day', start=2000, stop=2002, **kw).init()
    assert yearlike.t.npts == 731 # A bare number from 1900 on as the start is a calendar year, whatever the unit

    # The unit belongs to the model: the step does not change what its bare numbers mean, and typed values that disagree need unit=
    S, I, R = ss.states('S I R')
    typed = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init_prev=0.01, name='sir')
    daily, weekly = [ss.Sim(typed, stop=84, dt=dt, **kw).run() for dt in [ss.days(1), ss.weeks(1)]]
    assert weekly.unit == 'day' and daily.results.sir.I.max() > 1000 and weekly.results.sir.I.max() > 1000 # beta stays per day
    S, I, R = ss.states('S I R')
    with pytest.raises(TypeError, match='so the unit of the bare numbers mixed.beta is not known'):
        ss.Sim(ss.Disease(S >> I @ 0.3*I, I >> R @ ss.weeks(1), R >> S @ ss.days(100), name='mixed'), **kw).init()
    with pytest.raises(TypeError, match='mixed.rate'): # As is a bare number in a disease's death
        ss.Sim(ss.Disease(S >> I @ ss.perday(0.3)*I, I >> R @ ss.weeks(1), I >> ss.dead @ 0.01, name='mixed'), **kw).init()
    deaths = ss.death(rate=10) # But the deaths of everyone, as ss.Deaths reads them: per year, and above 1 per 1,000, with a note (not a warning: it is the usual spelling)
    assert ss.birth(rate=30).rate.raw == ss.peryear(0.03) and ss.birth(rate=ss.peryear(0.02)).rate.raw == ss.peryear(0.02)
    assert ss.birth(rate=ss.peryear(30)).rate.raw == ss.peryear(0.03) and ss.death(rate=ss.peryear(1.05)).rate.raw == ss.peryear(1.05/1000) # A typed rate is read as a bare number is: above 1 per year, per 1,000
    mixed = ss.Sim(ss.Disease(S >> I @ ss.perday(0.3)*I, I >> R @ ss.weeks(1), name='mixed'), demographics=deaths, **kw).init()
    assert mixed.diseases.mixed.pars.rate_deaths.raw == ss.peryear(0.01) and 'rate=10 is above 1 per year, so taken as per 1,000 people per year' in mixed.notes
    zero = ss.par(0.0, name='zero') # But 0 is 0 in any unit
    assert ss.Sim(ss.Disease(S >> I @ ss.perday(0.3)*I, I >> R @ ss.weeks(1), R >> S @ zero, name='mixed'), **kw).init().unit == 'day'

    # A rate and its duration say the same thing, whatever the unit
    I, R = ss.states('I R')
    a = ss.Disease(recovery=ss.Transition(I, R, dur=ss.weeks(1)), default_state='I')
    I, R = ss.states('I R')
    b = ss.Disease(recovery=ss.Transition(I, R, rate=ss.perday(1/7)), default_state='I')
    res = [ss.Sim(mod, stop=30, unit='day', **kw).run().diseases[0].results.I for mod in [a, b]]
    assert np.allclose(res[0], res[1])
    assert np.isclose(res[0][7], 10_000*np.exp(-1), rtol=1e-4) # After seven steps, a week

    # A duration as stop= with a start date is the length of the run; a rate of 0 never ends a dwell; and ages can be any distribution
    assert ss.Sim(make_sir(), start='2020-01-01', dur=ss.days(735), **kw).init().t.stop == ss.date('2020-01-01') + ss.days(735)
    I, R = ss.states('I R')
    assert ss.Sim(ss.Disease(recovery=ss.Transition(I, R, rate=0), default_state='I'), stop=10, n_agents=100, **kw).run().diseases[0].results.I[-1] == 100
    assert ss.Sim(make_sir(), people=ss.People(1000, age_data=ss.uniform(20, 30)), stop=2, **kw).init().people.age.min() >= 20

    # Times of entering a state are step numbers (ss.TimeArr), NaN if never
    sir = ss.Sim(make_sir(), stop=100, unit='day', n_agents=1000, **kw).run().diseases[0]
    assert isinstance(sir.ti_I, ss.TimeArr) and sir.ti_I.raw.dtype == np.float32 and sir.exit_I.raw.dtype == np.int8 # 4 bytes and 1
    infected = sir.ti_I.notnan.uids
    assert np.isnan(sir.ti_I[sir.S.uids]).all() and (sir.ti_I[infected] <= sir.ti).all() # Never infected is NaN, not a large negative number
    later = infected[sir.ti_I[infected] >= 0]
    assert np.allclose(sir.ti_I.years[later], sir.t.yearvec[sir.ti_I[later].astype(int)]) # The year at the start of the step of infection
    assert (sir.ti_I[infected] >= -1).all() and (sir.ti_I[infected] == -1).sum() == sir.results.I[0] # Those infectious at the start are dated as if infected in the step before the first, index -1
    with pytest.raises(AttributeError, match='sir.ti_R is not kept'): # The engine does not need ti_R, so it was not kept
        sir.ti_R.notnan
    class Recovered(ss.Analyzer): # A Starsim 3-style module that reads it, which needs it kept
        def step(self):
            self.n = self.sim.diseases.sir.ti_R.isclose(self.ti).sum() # A move reads as the step it happened in (see ss.TimeArr)
    sim = ss.Sim(make_sir(), stop=100, unit='day', n_agents=1000, analyzers=Recovered(), keep_times=True, **kw).run() # For this sim (ss.options.keep_times for every sim)
    assert np.array_equal(sim.diseases.sir.results.I, sir.results.I) and sim.analyzers[0].n == sim.results.sir.recovery[-1] # The same run, and the recoveries of the last step
    assert sim.diseases.sir.recovery.n_times.values.sum() == sim.results.sir.recovery.sum() # And the history of every transition, to read after the run
    S, I, R = ss.states('S I R')
    seeded = ss.Sim(ss.Disease(S >> I @ 0*I, I >> R @ ss.days(5), name='seeded'), stop=30, unit='day', n_agents=100, **kw).init()
    seeded.diseases.seeded.S[ss.uids(np.arange(10))] = False
    seeded.diseases.seeded.I[ss.uids(np.arange(10))] = True
    assert seeded.run().results.seeded.I[-1] == 0 # Put in a state with no time of entering, they enter it now and leave it (they stayed for ever before)

    # With no unit, a probability is per step, and times with no calendar are numbers
    I, R = ss.states('I R')
    c = ss.Disease(recovery=ss.Transition(I, R, rate=ss.prob(0.05)), default_state='I')
    assert np.isclose(ss.Sim(c, stop=10, **kw).run().diseases[0].results.I[10], 10_000*0.95**10) # After ten steps
    df = ss.Sim(make_sir(), stop=10, **kw).run().results.sir.to_df()
    assert df.time.iloc[1] == 1 # Not a date in year 1

    # A number times a probability scales its rate: 0.275 of people a year, not 0.16; a probability of 1 is an infinite rate (so still 1 when scaled), which moves everyone in the step, under every solver
    S, I = ss.states('S I')
    res = ss.Sim(ss.Disease(S, I, onset=S >> I @ ss.probperyear(0.8)*0.2, name='d'), total_pop=10_000, solver='ode', start=2000, stop=2001, **kw).run().results.d
    assert np.isclose(res.I[-1], 10_000*(1 - 0.2**0.2), rtol=1e-3)
    for solver in ['abm', 'ode', 'binomial']:
        res = ss.Sim(ss.Disease(S, I, onset=S >> I @ ss.probperyear(1)*0.8, name='d'), n_agents=1000, total_pop=1000, solver=solver, start=2000, stop=2001, **kw).run().results.d
        assert res.S[1] == 0 and res.I[1] == res.S[0] + res.I[0], solver
    return sim


@sc.timer()
def test_solvers(do_plot=False):
    sc.heading('Testing the stochastic solvers by name, and the ODE against known answers')
    # The exact stochastic solver, at a smaller size, stays exact however many events a run takes; tau-leaping is asked for by name
    sim = ss.Sim(make_sir(), total_pop=2000, stop=100, solver='gillespie', **kw).run()
    assert sim.results.sir.R[-1] > 1000 and isinstance(sim.solver, ss.solvers.gillespie)
    assert isinstance(ss.Sim(make_sir(), total_pop=1e8, solver='gillespie', **kw).init().solver, ss.solvers.gillespie)
    tau = ss.Sim(make_sir(), total_pop=1e8, stop=100, solver='tau', **kw).run()
    assert isinstance(tau.solver, ss.solvers.tau) and tau.results.sir.R[-1] > 5e7
    for old in ['sde', 'stochastic', 'ctmc']: # Not solvers
        with pytest.raises(ValueError, match='gillespie'):
            ss.Sim(make_sir(), solver=old, **kw).init()
    with pytest.raises(ValueError):
        ss.Sim(make_sir(), solver=ss.solvers.ode(integrator='LSODA'), **kw).init()

    # Compare with the known final size of an SIR epidemic
    sim = ss.Sim(make_sir(), total_pop=100_000, dt=0.25, stop=100, solver='ode', **kw).run()
    final = sim.results.sir.R[-1]/100_000
    expected = 0.9405 # The root of z = 1 - 0.99*exp(-3*z), with R0 = 3
    assert np.isclose(final, expected, atol=0.005), f'Expected the final size to be {expected:n}, not {final:n}'

    # The default ODE follows a fast epidemic at a daily step (K&R program 2.1): the substeps follow how fast the infected grow, not only how fast they recover
    from scipy.integrate import solve_ivp
    exact = solve_ivp(lambda t, y: [-1.4247*y[0]*y[1], 1.4247*y[0]*y[1] - 0.14286*y[1]], (0, 70), [1-1e-6, 1e-6], t_eval=np.arange(71.), method='DOP853', rtol=1e-12, atol=1e-15).y[1]
    S, I, R = ss.states(3)
    kr21 = ss.Disease(infection=S >> I @ ss.perday(1.4247) * I, recovery=I >> R @ ss.perday(0.14286), init=dict(I=1e-6), name='sir')
    n_I = ss.Sim(kr21, unit='day', stop=70, total_pop=1.0, **kw).run().results.sir.I.values
    assert np.abs(n_I - exact).max()/exact.max() < 1e-4 # 2.5% off before, with one RK4 step a day

    if do_plot:
        sim.plot()
    return sim


@sc.timer()
def test_auto():
    sc.heading('Testing how the solver is chosen')
    sim = ss.Sim(make_sir(), stop=10, **kw).init()
    assert isinstance(sim.solver, ss.solvers.ode)
    agents = ss.Sim(make_sir(), stop=10, n_agents=1000, **kw)
    assert isinstance(agents.init().solver, ss.solvers.abm)
    assert isinstance(agents.new(n_agents=None).init().solver, ss.solvers.ode) # None goes back to the default
    msim = ss.parallel(ss.Sim(make_sir(), stop=10, solver='binomial', **kw), n_runs=2)
    assert 'sir' in msim.results.keys() and msim.results['sir'] is msim.results.sir
    assert isinstance(ss.Sim(make_sir(), networks='random', stop=10, **kw).init().solver, ss.solvers.abm)
    assert isinstance(ss.Sim(make_sir(), demographics=ss.Births(), start=2000, stop=2010, **kw).init().solver, ss.solvers.abm)

    # A dwell time that is not exponential is run as Erlang stages, unless approximations='error'
    S, I, R = ss.states('S I R')
    sir = ss.Disease(
        infection = ss.Transition(S, I, rate=0.3 * I),
        recovery  = ss.Transition(I, R, dur=ss.lognormal(mean=10, std=5)),
        init_prev = 0.01,
    )
    sim = ss.Sim(sir, stop=100, **kw).run()
    assert isinstance(sim.solver, ss.solvers.ode)
    assert len(sim.diseases[0].engine.state_comps['I']) == 4 # (10/5)² stages
    assert any('Erlang' in note for note in sim.notes)
    assert isinstance(ss.Sim(sir, stop=100, approximations='error', **kw).init().solver, ss.solvers.abm)
    with pytest.raises(ss.CapabilityError):
        ss.Sim(sir, stop=100, approximations='error', solver='ode', **kw).init()

    # A distribution on beta varies between agents, so agents are chosen; a compartmental run uses its mean
    S, I, R = ss.states('S I R')
    spread = ss.Disease(infection=ss.Transition(S, I, rate=ss.lognormal(0.3, 0.3) * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
    auto = ss.Sim(spread, stop=100, **kw).run()
    assert isinstance(auto.solver, ss.solvers.abm) and 'distribution' in auto.solver_reason
    fixed = ss.Sim(make_sir(), stop=100, solver='abm', **kw).run()
    assert auto.results.sir.infection.sum() < 0.9*fixed.results.sir.infection.sum() # Varied susceptibility gives fewer infections
    ode = ss.Sim(spread, stop=100, solver='ode', **kw).run()
    assert np.allclose(ode.results.sir.I, ss.Sim(make_sir(), stop=100, solver='ode', **kw).run().results.sir.I)
    assert any('distribution across people' in note for note in ode.notes)
    with pytest.raises(ss.CapabilityError):
        ss.Sim(spread, stop=100, approximations='error', solver='ode', **kw).init()
    with pytest.raises(ss.CapabilityError, match='approximations="error": RandomNet approximated'): # A network has no exact compartmental form, so the ODE approximates it
        ss.Sim(make_sir(), networks='random', stop=10, approximations='error', solver='ode', **kw).init()

    # A mixing pool of some people needs agents
    pool = ss.MixingPool(source=ss.person.age < 15, target=ss.person.age >= 15)
    assert isinstance(ss.Sim(make_sir(), networks=pool, stop=10, **kw).init().solver, ss.solvers.abm)
    with pytest.raises(ss.CapabilityError, match='source= or target='):
        ss.Sim(make_sir(), networks=pool, solver='ode', stop=10, **kw).init()

    # A module written as code cannot run as compartments
    with pytest.raises(ss.CapabilityError):
        ss.Sim(make_sir(), demographics=ss.Births(), solver='ode', start=2000, stop=2010, **kw).init()
    with pytest.raises(ss.CapabilityError): # Nor can a model with nothing declared
        ss.Sim(diseases=ssl.AgentSIR(), networks='random', solver='ode', **kw).init()
    return sim


@sc.timer()
def test_describe():
    sc.heading('Testing describe(), diagram(), equations(), R0(), and the transition table')
    S, I, R = ss.states(3)
    sir = ss.Disease(
        infection = ss.Transition(S, I, rate=0.05 * I),
        recovery  = ss.Transition(I, R, dur=10),
    )
    sim = ss.Sim(sir, stop=100, **kw)
    text = sim.describe()
    print(text)
    for string in ['10,000 people [guessed]', 'time is in timesteps',
                   'rate = 0.05 × I.n / N', 'exponential(mean = 10 steps)', 'homogeneous', 'every step: sir']:
        assert string in text, f'Expected "{string}" in describe()'
    assert '= 0.0952 per step' in sim.describe() # The probability of recovering in a step, 1 - exp(-1/10)
    S, I, R = ss.states(3)
    density = ss.Sim(ss.Disease(S >> I @ ss.perday(3e-5)*I.n, I >> R @ ss.days(10), init_n=dict(I=10)), **kw).init()
    assert 'mixing' not in density.describe() # Nothing is contact-mediated, so there is no mixing to describe (it said "frequency-dependent")
    assert 'infected: I ' in density.describe() # Density-dependent transmission is a transmission: I is its source, and R is not infected (it counted R)
    assert np.isclose(density.diseases[0].R0(), 3e-5*(10_000 - 10)*10) # beta × the susceptible × the time infectious (it failed, "Singular matrix"; then counted the 10 infected as susceptible)
    share = ss.Sim(ss.Disease(S >> I @ ss.perday(0.3)*I.frac, I >> R @ ss.days(10), init_n=dict(I=10)), **kw).init() # A share of everyone (as with a contact matrix): I is the source here too
    assert 'infected: I ' in share.describe() and np.isclose(share.diseases[0].R0(), 3, rtol=1e-6) # It counted R, and R0 failed ("Singular matrix")
    S, I, R = ss.states('S I R')
    aged = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir').stratify(age=[0, 15])
    births = ss.Sim(aged, demographics=[ss.birth(rate=ss.peryear(0.02)), ss.death(rate=ss.peryear(0.02))], networks=ss.contacts([[3, 1], [1, 2]]), unit='day', total_pop=10_000, **kw).init().describe()
    assert '0.02 per year × (S + I + R).n' in births # A rate in another unit than the sim's keeps its own (it was "0.02", under "in total per day")
    assert 'literally' not in births # Births are in total, so the note about contacts does not apply to them
    seasonal = ss.Sim(ss.Disease(S >> I @ ss.seasonal(mean=ss.perday(17/13), amplitude=0.1, peak=0.25) * I, I >> R @ ss.perday(1/13), init=dict(I=1e-4), name='sir'), unit='day', total_pop=1.0, **kw).init()
    assert 'beta = 1.30769 per day × (1 + 0.1 cos(2π(t − 0.25)))' in seasonal.describe() # As the book writes it (it was "seasonal(mean=1.3076923076923077, ...)")
    assert 'beta = perday(1.30769) × (1 + 0.1 cos(2π(t − 0.25))), with t in years' in seasonal.diseases.sir.equations() # What the symbol stands for
    assert 'init' in ss.Sim(ss.SIR(), stop=50, **kw).describe() # init_prev is an ss.par
    S, I, R = ss.states(3)
    beta, gamma = ss.par(0.5, doc='transmission rate', source='Smith 2020, table 2'), ss.par(ss.perday(0.1), source='expert opinion')
    documented = ss.Sim(ss.Disease(S >> I @ beta*I, I >> R @ gamma, name='sir'), unit='day', **kw).describe()
    for string in ['beta = 0.5', 'transmission rate', 'source: Smith 2020, table 2', 'gamma = 0.1 per day', 'source: expert opinion']: # Each parameter's doc= and source=, under it
        assert string in documented, f'Expected "{string}" in describe()'
    recovery = ss.par(ss.perday(0.1), label='Recovery rate', doc='per day')
    assert repr(recovery) == "par(perday(0.1), label='Recovery rate', doc='per day')" # A label, and a doc that is short
    assert 'recovery (Recovery rate) = 0.1 per day' in ss.Sim(ss.Disease(S >> I @ beta*I, I >> R @ recovery, name='sir'), unit='day', **kw).describe()
    ss.SIR().equations() # Before the disease is in a sim

    assert sir.diagram('text').splitlines() == ['flowchart LR', '    S(["S"])', '    I(["I"])', '    R(["R"])', '    S -.->|"infection: 0.05 × I"| I', '    I -->|"recovery: 10"| R'] # Values, not names (it was "recovery: dur = dur")
    assert '```mermaid' in sir.diagram('text')._repr_markdown_() # Drawn in a notebook
    assert len(sir.diagram(do_show=False).axes[0].patches) >= 5 # By default, drawn with Matplotlib: three states and two arrows

    eqs = sir.equations()
    print(eqs)
    assert eqs.splitlines() == ['dS/dt = -beta * S * I / N', 'dI/dt =  beta * S * I / N - I / dur_I', 'dR/dt =  I / dur_I']
    S, I, R = ss.states('S I R')
    fatal = ss.Disease(S >> I @ 0.4*I, ss.Transition(I, {R: None, ss.dead: 0.002}, dur=ss.days(8)), name='fatal')
    assert 'dR/dt =  (1 - 0.002) * I / dur' in fatal.equations() # A branch to ss.dead, given in a list, takes its share (it was "I / dur")
    text = fatal.describe()
    assert 'null1' in fatal.diagram('text') and 'I -> R | (removed)' in text # ss.dead shown as leaving the model (it was "I -> ")
    assert 'beta = 0.4 per day' in text # Outside a sim, in the unit of its typed values (it was "per step")
    assert '8 days, p = 0.002"| null1' in fatal.diagram('text') # Each branch with its share
    S, I, R = ss.states(3)
    beta = ss.par(0.3, bounds=[0.1, 1])
    infection = S >> I @ beta*I
    cases = ss.outcome(0.5*infection, delay=ss.days(4))
    text = ss.Disease(infection, I >> R @ ss.erlang(k=4, mean=ss.days(8)), cases, name='sir').stratify(risk={'low': 0.8, 'high': 0.2}).describe()
    for string in ['beta = 0.3 per day, between 0.1 and 1', 'erlang(k=4, mean=8 days)', 'outcome: 0.5 × infection, after 4 days (exponential)']: # Bounds, a distribution as given, and an outcome's definition
        assert string in text, f'Expected "{string}" in describe()'
    table = repr(sir.transitions)
    print(table)
    assert 'transmission, from I' in table
    assert ss.expressions.format_expr(sir.graph.transitions.infection.rate, exact=True) == 'beta * I'

    # R0 from the next-generation matrix: beta × the mean time infectious, with a competing death; the contact matrix's largest eigenvalue; and through a reservoir
    assert np.isclose(sir.R0(), 0.5)
    S, I, R = ss.states('S I R')
    immune = ss.Sim(ss.Disease(S >> I @ 1.9*I, I >> R @ ss.days(8), init=dict(R=0.9, I=1e-3), name='measles'), unit='day', solver='ode', verbose=0).init()
    assert np.isclose(immune.diseases.measles.R0(), 15.2) # With no immunity, whoever starts immune (it was 1.52, R at the start)
    assert 'R = 1.51 at the start, with 10% susceptible' in '\n'.join(immune._start_lines())
    S, E, I, R = ss.states('S E I R')
    seir = ss.Disease(infection=ss.Transition(S, E, rate=0.3 * I), onset=ss.Transition(E, I, dur=5),
                      recovery=ss.Transition(I, R, dur=10), death=ss.death(I, rate=0.05))
    assert np.isclose(seir.R0(), 0.3/(0.1 + 0.05))
    C = np.array([[10, 3], [3, 5]])
    S, I, R = ss.states('S I R')
    aged = ss.Sim(ss.Disease(infection=ss.Transition(S, I, rate=0.03 * I), recovery=ss.Transition(I, R, dur=10)), networks=ss.contacts(C, ages=[0, 20]), **kw)
    assert np.isclose(aged.init().diseases[0].R0(), 0.3*np.linalg.eigvals(C).max()) # With the mixing of its sim
    assert 'C[i, j] * I[j] / N[j]' in aged.diseases[0].equations() # And its equations say so, with the matrix
    cholera = ss.Sim(make_cholera(), total_pop=10_000, unit='day', **kw)
    assert np.isclose(cholera.init().diseases[0].R0(), 0.5/1e5*9_990*10*5*30) # Infection × shedding × time infectious × time in the water, with the 0.1% infected at the start left out

    # At the disease-free state: the seeded infected are not susceptible; a constant inflow sets the people; and a chain of states of its own keeps its people
    S, I, R = ss.states('S I R')
    seeded = ss.Sim(ss.Disease(S >> I @ ss.perday(0.1)*I.n, I >> R @ ss.days(1), init_n=dict(I=1)), total_pop=21, **kw)
    assert np.isclose(seeded.init().diseases[0].R0(), 0.1*20) # β(N - 1)/γ (it was βN/γ = 2.1)
    S, I, R = ss.states('S I R')
    H = ss.State(rel_sus=2) # At high risk
    inflow = ss.Disease(S + H >> I @ ss.perday(0.2)*I, I >> R @ ss.days(5), ss.unborn >> S @ ss.perday(8), ss.unborn >> H @ ss.perday(2), ss.death(rate=ss.perday(0.001)),
                        init_n=dict(S=990, I=10), name='inflow')
    split = ss.Sim(inflow, total_pop=1000, unit='day', **kw).init()
    assert np.isclose(split.diseases.inflow.R0(), 0.2*(0.8 + 0.2*2)/(0.2 + 0.001)) # At the split of the inflows, 80/20 (it was all in S)
    assert '2 * beta * H * I / N' in inflow.equations() and 'H susceptible, rel_sus = 2' in inflow.describe() # And rel_sus is shown (it was left out)
    (S1, I1, R1), (S2, I2, R2) = [ss.states(f'S{h} I{h} R{h}', people=True) for h in [1, 2]] # Two host species, each infected by the other; nothing links them
    hosts = ss.Disease(S1 >> I1 @ 0.1*I2.n/1000, I1 >> R1 @ 0.1, S2 >> I2 @ 0.4*I1.n/1000, I2 >> R2 @ 0.1, init_n=dict(S1=500, S2=500, I1=1))
    assert np.isclose(ss.Sim(hosts, total_pop=1001, **kw).init().diseases[0].R0(), np.sqrt(0.1*0.4*500*500/1000**2*100)) # Each host's people where they are (it was all of them in S1)
    return sim


@sc.timer()
def test_load():
    sc.heading('Testing ss.save() and ss.load()')
    S, I, R = ss.states(3)
    W = ss.State(people=False) # Bacteria in the water
    infection = S >> I @ ss.perday(0.5) * W.n / (W.n + 1e5)
    cholera = ss.Disease(S, I, R, W,
        infection = infection,
        recovery  = ss.Transition(I, {R: 0.9, S: 0.1}, dur=ss.lognormal(5, 2, unit='days')),
        shedding  = ss.outside >> W @ ss.par(10, bounds=[5, 20], name='shed') * I.n,
        decay     = W >> ss.dead @ ss.days(30),
        cases     = ss.outcome(0.5*infection, delay=ss.days(3)),
        prev      = I/(S + I + R),
        init_n    = dict(I=5),
    )
    cholera.stratify(age=[0, 15])
    sim = ss.Sim(cholera, connectors=ss.modify(cholera.recovery, multiply=ss.by(age=[1.5, 1])), unit='day', start='2020-01-01', stop='2020-07-01', total_pop=1000, solver='gillespie', **kw)
    sim.stratify(risk={'low': 0.8, 'high': 0.2})
    sim.save('temp.sim') # Before it has run
    sim2 = ss.load('temp.sim').run()
    assert np.array_equal(sim2.results.cholera.cases.values, sim.run().results.cholera.cases.values)

    # A sim that has run can be saved and loaded under every solver
    for solver in ['ode', 'binomial', 'gillespie', 'abm']:
        pars = dict(n_agents=500) if solver == 'abm' else {}
        sim = ss.Sim(make_sir(), stop=50, solver=solver, **pars, **kw).run()
        sim.save('temp.sim')
        assert np.array_equal(ss.load('temp.sim').results.sir.I, sim.results.sir.I), solver
    sc.rmpath('temp.sim')

    # Rates are written with the parentheses they need, so they are read back the same
    S, I, W = ss.State(name='S'), ss.State(name='I'), ss.State(name='W')
    rate = 1 - (1 - 0.5*I.frac)**3 / (2*W.n) - (I.n - W.n)
    assert ss.expressions.format_expr(rate) == '1 - (1 - 0.5 * I.frac) ** 3 / (2 * W.n) - (I.n - W.n)'
    return sim2


@sc.timer()
def test_transfer():
    sc.heading('Testing vaccination as a stratification plus a transfer')
    sims = sc.obj()
    for solver in ['ode', 'abm']:
        sir = make_sir()
        sir.pars.beta.value = 0.2
        sir.stratify(risk={'low': 0.8, 'high': 0.2}, vaccinated={False: 1, True: 0}, vaccination=ss.transfer(dict(vaccinated=True), rate=ss.perday(0.02), states=['S']))
        by_group = ss.modify(sir.infection, multiply=ss.by(risk=[1.0, 3.0])*ss.by(vaccinated=[1.0, 0.1]))
        pars = dict(n_agents=20_000) if solver == 'abm' else {}
        sims[solver] = ss.Sim(sir, connectors=by_group, total_pop=100_000, unit='day', stop=150, solver=solver, **pars, **kw).run()
    ode, abm = sims.ode.results.sir, sims.abm.results.sir
    assert ode.vaccination.cum[-1] > 50_000
    assert ode.infection.select(risk='high').sum()/20_000 > ode.infection.select(risk='low').sum()/80_000 # The high-risk group has the higher attack rate
    assert ode.infection.select(vaccinated=True).sum() < 0.1*ode.infection.sum() # The disease never mentions vaccination
    for key in ['infection', 'vaccination']:
        a, o = abm[key].cum[-1], ode[key].cum[-1]
        assert np.isclose(a, o, rtol=0.2), f'Expected agents to match the ODE for {key}, but {a:n} vs {o:n}' # One run, at a daily step: a difference of order dt (agents are 10% below the ODE over seeds), plus the noise of one run (5%); this seed moved from -9% to -16% when the random streams moved with the time index during a step

    # The unvaccinated model has a larger epidemic
    plain = make_sir()
    plain.pars.beta.value = 0.2
    plain.stratify(risk={'low': 0.8, 'high': 0.2})
    base = ss.Sim(plain, connectors=ss.modify(plain.infection, multiply=ss.by(risk=[1.0, 3.0])), total_pop=100_000, unit='day', stop=150, **kw).run()
    assert base.results.sir.infection.cum[-1] > 1.2*ode.infection.cum[-1]

    # A dimension that people are moved along, with no split given, starts with everyone at its first level
    staged = make_sir().stratify(stage=['early', 'late'], progress=ss.transfer(dict(stage=('early', 'late')), rate=ss.perday(0.01)))
    assert np.allclose(ss.Sim(staged, unit='day', stop=10, **kw).init().groups.strats[0].fractions, [1, 0])

    # A transfer along a dimension the model does not have is an error
    with pytest.raises(KeyError):
        I, R = ss.states('I R')
        ss.Sim(ss.Disease(recovery=ss.Transition(I, R, dur=3), switch=ss.transfer(dict(risk=('low', 'high')), rate=0.1)), **kw).init()
    # A transfer given to stratify() registers its parameters too, and a transition can be called "stop"
    c = ss.par(0.0, name='c')
    S, I = ss.states(2)
    later = ss.Disease(inf=S >> I @ 0.1*I, stop=I >> S @ 0.1, init=dict(I=0.01), name='later')
    later.stratify(v=[False, True], switch=ss.transfer(dict(v=True), rate=c*0.2))
    assert 'c' in later.pars and 'stop' in later.transitions
    assert ss.Sim(later, n_agents=500, stop=10, pars=dict(c=1), **kw).run().results.later.switch.sum() > 0

    # A transfer's rate can hold a parameter, set by name, and can be modified
    for solver,agents in [('ode', {}), ('abm', dict(n_agents=2000))]:
        c = ss.par(0.0)
        S, I = ss.states(2)
        dis = ss.Disease(inf=S >> I @ 0.1*I, switch=ss.transfer(dict(v=True), rate=c*0.2), init=dict(I=0.01), name='dis')
        dis.stratify(v=[False, True])
        assert ss.Sim(dis, stop=20, solver=solver, pars=dict(c=1), **agents, **kw).run().results.dis.switch.sum() > 0
        assert ss.Sim(dis, stop=20, solver=solver, pars=dict(c=1), interventions=ss.modify(dis.switch, multiply=0), **agents, **kw).run().results.dis.switch.sum() == 0

    # A transfer's dur= distribution is drawn for each person: as agents exactly, as compartments by Erlang stages
    S, I = ss.states(2)
    wane = ss.Disease(inf=S >> I @ 0*I, init=dict(I=0.0), name='wane')
    wane.stratify(rpr={False: 0, True: 1}, revert=ss.transfer(dict(rpr=False), dur=ss.lognormal(ss.years(15), ss.years(5))))
    s2 = np.log(1 + (5/15)**2)
    lognorm = stats.lognorm.cdf([8, 15], np.sqrt(s2), scale=15*np.exp(-s2/2))
    erlang = stats.gamma.cdf([8, 15], 9, scale=15/9) # 9 stages for a CV of 1/3
    for solver,expected,agents in [('abm', lognorm, dict(n_agents=5000)), ('ode', erlang, {})]:
        sim = ss.Sim(wane, unit='year', dt=0.5, start=0, stop=16, solver=solver, **agents, **kw).run()
        n = sim.results.wane.S.by('rpr').values
        moved = n[[16, 30], 0]/n[0].sum()
        assert np.allclose(moved, expected, atol=0.025), f'{solver}: {moved} moved by years 8 and 15, against {expected}' # Not the 0.41 and 0.63 of a rate of one over the mean
    yearly = ss.Disease(inf=S >> I @ 0*I, init=dict(I=0.0), name='yearly').stratify(school=[0, 1, 2, 3], year=ss.transfer(increment='school', dur=ss.constant(ss.years(1))))
    n = ss.Sim(yearly, unit='year', dt=0.25, start=0, stop=2.5, solver='abm', n_agents=200, **kw).run().results.yearly.S.by('school').values
    assert np.array_equal(n[[3, 4, 7, 8]].argmax(axis=1), [0, 1, 1, 2]) and n[[3, 4, 7, 8]].max(axis=1).min() == 200 # Everyone up a year on each birthday of the run, exactly

    # To one of several levels, with p=
    for solver,agents in [('abm', dict(n_agents=5000)), ('ode', {})]:
        fp = ss.Disease(inf=S >> I @ 0*I, init=dict(I=0.0), name='fp')
        fp.stratify(method={'none': 1, 'pill': 0, 'iud': 0}, start=ss.transfer(dict(method=('none', ['pill', 'iud'])), p=[0.6, None], dur=ss.lognormal(ss.years(2), ss.years(1))))
        n = ss.Sim(fp, unit='year', dt=0.25, start=0, stop=10, solver=solver, **agents, **kw).run().results.fp.S.by('method').values[-1]
        assert np.allclose(n/n.sum(), [0, 0.6, 0.4], atol=0.02), f'{solver}: {n/n.sum()}'
    with pytest.raises(ValueError, match='rate='): # One level to move to
        ss.transfer(dict(v=True), p=0.5, rate=0.1)
    return sims


def test_lifetable_and_seeding():
    sc.heading('Testing deaths from a life table, and seeding a number of people on a schedule')
    ages = np.arange(101)
    dx = np.clip(np.gradient(1 - np.exp(-(ages/80.0)**6)) + 1e-4, 0, None) # Most deaths in old age
    table = ss.lifetable(ages, dx)
    def make(strat=False):
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ 0.3*I.frac, I >> R @ ss.days(8), death=ss.death(lifetable=table), init=dict(I=0), name='sir')
        return sir.stratify(age=[0, 20, 40, 60, 70, 80, 90]) if strat else sir

    # As agents, the death rate of each person's age band of the table, as ss.Deaths(lifetable=) takes it, drawn as each person's age at death once
    ppl = ss.People(20_000, age_data=np.array([np.arange(100), np.exp(-0.02*np.arange(100))]).T)
    sim = ss.Sim(make(), people=ppl, unit='year', dt=1, stop=3, verbose=0).init()
    assert isinstance(sim.solver, ss.solvers.abm) # Without a stratification by age, a life table needs agents
    age = np.asarray(sim.people.age.raw[sim.people.auids])
    expected = np.sum(-np.expm1(-table.hazards(ages)[np.minimum(age.astype(int), len(ages) - 1)])) # The band's rate, over the year
    res = sim.run().results.sir
    assert np.isclose(res.death[0], expected, rtol=0.1) and np.isclose(res.S[0] - res.S[1], res.death[0]) # Deaths in the first year, as the table gives them
    with pytest.raises(ss.CapabilityError):
        ss.Sim(make(), unit='year', stop=3, solver='ode', verbose=0).init()
    sir = make()
    with pytest.raises(ss.CapabilityError, match='drawn once'): # Drawn once, so the rate cannot change during the run
        ss.Sim(sir, interventions=ss.modify(sir.death, multiply=2), people=ss.People(1000), unit='year', stop=3, verbose=0).run()
    ode = ss.Sim(make(strat=True), unit='year', stop=3, solver='ode', verbose=0).run() # Stratified by age, it is a death rate for each age group
    assert ode.results.sir.death[1] > 0

    # Seeding: three people infected every 30 days for a year, chosen from everyone susceptible, as LASER's importation
    for solver in ['ode', 'abm']:
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ ss.perday(0.05)*I.frac, I >> R @ ss.days(8), init=dict(I=0), name='sir')
        seeds = ss.Event(S >> I, when=ss.schedule(every=ss.days(30), start=ss.days(30), stop=ss.days(365)), n=3, name='importation')
        delivered = ss.Sim(sir, interventions=seeds, unit='day', stop=400, solver=solver, n_agents=10_000, total_pop=10_000, verbose=0).run().results.importation.delivered.values
        assert list(np.flatnonzero(delivered)[:3]) == [30, 60, 90] and np.allclose(delivered[delivered > 0], 3) # The steps that start at 30, 60, 90
    return sim


def test_cross_disease_rate():
    sc.heading('Testing a rate that counts another disease\'s state')
    def final(cross, solver):
        S, I, R = ss.states('S I R')
        syph = ss.Disease(infection=ss.Transition(S, I, rate=0.0 * I), recovery=ss.Transition(I, R, rate=ss.peryear(1e-12)), init_n=dict(I=2000), name='syph') # 10% infected throughout
        beta = 0.15*(1 + 5*syph.I.n/20_000) if cross else 0.225 # So the same
        S, I, R = ss.states('S I R')
        hiv = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, dur=10), init_n=dict(I=100), name='hiv')
        kw = dict(n_agents=20_000) if solver == 'abm' else {}
        return ss.Sim([syph, hiv], total_pop=20_000, stop=10, solver=solver, verbose=0, **kw).run().results.hiv.I[-1]
    for solver in ['ode', 'abm']:
        assert np.isclose(final(True, solver), final(False, solver), rtol=1e-9) # syph.I is syphilis's, not HIV's own I

    # The same for a condition: where=a.C is the other disease's C, whether or not this one has a C too
    def canceled(names, solver):
        S, I, C = ss.states('S I C')
        a = ss.Disease(infection=S >> I @ ss.peryear(2)*I, cancer=I >> C @ ss.years(2), init_prev=0.2, name='a')
        S, I, C, X = ss.states(names)
        b = ss.Disease(infection=S >> I @ ss.peryear(2)*I, cancer=I >> C @ ss.years(2), cancel=ss.Transition(I, X, rate=ss.peryear(4), who=a.C), init_prev=0.2, name='b')
        return ss.Sim([a, b], n_agents=2000, total_pop=2000, stop=2005, solver=solver, verbose=0).run().results.b.cancel.sum()
    for solver in ['ode', 'abm']:
        assert np.isclose(canceled('S I C X', solver), canceled('S I D X', solver), rtol=1e-9) # Was 0 (b's own C) with agents, and an error with the ODE

    # A state of the other disease read before that disease is declared: a would keep an empty state of its own, so it is refused
    Sa, Ia, Sb, Ib, Rb = ss.states('Sa Ia Sb Ib Rb')
    a = ss.Disease(infection=Sa >> Ia @ ss.peryear(1.5)*Ia*(1 - 0.9*Rb.frac), name='a')
    with pytest.raises(ValueError):
        ss.Disease(infection=Sb >> Ib @ ss.peryear(1.5)*Ib, recovery=Ib >> Rb @ ss.years(1), name='b')
    return


def test_first_state():
    sc.heading('Testing that the people left over start in the first state: first=, else the state named S, else the first made')
    for solver in ['ode', 'abm']:
        R, I, S = ss.states(3)
        make = lambda **kw: ss.Disease(infection=S >> I @ 0.3*I.frac, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir', **kw)
        assert make().graph.default_state.name == 'S' # By its name, whatever order the states were made in
        sim = ss.Sim(make(), total_pop=10_000, n_agents=10_000, stop=1, unit='day', solver=solver, verbose=0).run()
        assert sim.results.sir.S[0] == 9900 and sim.results.sir.R[0] == 0
        assert ss.Sim(make(default_state='R'), total_pop=10_000, n_agents=10_000, stop=1, unit='day', solver=solver, verbose=0).run().results.sir.R[0] == 9900 # Or given

    # With no S and no first=, the first state made, whatever the order of the transitions (it raised), and describe() says it is a guess
    B, A = ss.states('B A')
    sim = ss.Sim(ss.Disease(prog=A >> B @ 0.1, back=B >> A @ 0.1, name='d'), stop=2, verbose=0).run()
    assert sim.diseases.d.graph.default_state.name == 'B' and sim.results.d.B[0] == 10_000
    assert 'the first state made' in sim.describe()

    # The first state given, whatever the order of the transitions
    S, I, V = ss.states('S I V')
    trs = dict(infection=S >> I @ ss.perday(0.3)*I, vax=S >> V @ ss.perday(0.01), breakthrough=V >> I @ ss.perday(0.1)*I)
    sir = ss.Disease(**{k:trs[k] for k in reversed(trs)}, init_n=dict(I=100), name='sir', default_state='V')
    sim = ss.Sim(sir, total_pop=10_000, stop=10, unit='day', solver='ode', verbose=0)
    assert list(sir.graph.transitions.keys()) == ['breakthrough', 'vax', 'infection'] # In the order given
    assert sim.run().results.sir.V[0] == 9900
    return


@sc.timer()
def test_seeded_stay():
    sc.heading('Testing that people infectious at the start have been so for the first step, as if infected in the step before')
    for solver in ['ode', 'binomial', 'abm']: # A seeded case with a fixed stay of one step leaves after it, as in the Reed-Frost model (before, it stayed and infected for two)
        S, I, R = ss.states(3)
        rf = ss.Disease(infection=S >> I @ ss.perday(0.0)*I.n, recovery=I >> R @ ss.constant(ss.days(1)), init_n=dict(I=10), name='rf')
        sim = ss.Sim(rf, unit='day', stop=5, total_pop=100, n_agents=100, solver=solver, verbose=0).run()
        assert sim.results.rf.I[0] == 10 and sim.results.rf.I[1] == 0
    for solver in ['binomial', 'abm']: # So the seeds infect for one generation, as the people they infect do
        S, I, R = ss.states(3)
        rf = ss.Disease(infection=S >> I @ ss.perday(0.001)*I.n, recovery=I >> R @ ss.constant(ss.days(1)), init_n=dict(I=100), name='rf')
        res = ss.Sim(rf, unit='day', stop=2, total_pop=10_000, n_agents=10_000, solver=solver, verbose=0).run().results.rf
        assert res.I[1] == res.infection[0] # Only the new cases of the first step are left after it
    # On a network too, a seed with a one-step stay infects in the first step only, as the people it infects do in theirs (before, never; by a rate, see the loop above)
    S, I, R = ss.states('S I R')
    sir = ss.Disease(S, I, R, infection=S >> I @ ss.perday(0.1)*I, recovery=I >> R @ ss.constant(ss.days(1)), init_n=dict(I=100), name='sir')
    sim = ss.Sim(sir, networks=ss.RandomNet(), unit='day', stop=4, n_agents=10_000, analyzers=ss.infection_log(), verbose=0).run()
    log = sim.analyzers[0].to_df()
    by_seeds = log[log.source.isin(log.target[log.source == -1])]
    assert len(by_seeds) and (by_seeds.t == 0).all(), f'seeds infected on days {sorted(set(by_seeds.t))}'
    return


@sc.timer()
def test_disconnected_groups():
    sc.heading('Testing two risk groups written as separate states, with no transition between them')
    def make():
        SH, IH, SL, IL = ss.states('SH IH SL IL')
        sis = ss.Disease(inf_H=SH >> IH @ ss.perday(10)*IH.frac + ss.perday(0.1)*IL.frac, inf_L=SL >> IL @ ss.perday(0.1)*IH.frac + ss.perday(1)*IL.frac,
                         rec_H=IH >> SH @ ss.perday(1), rec_L=IL >> SL @ ss.perday(1), init=dict(SH=0.2-1e-5, IH=1e-5, SL=0.8-1e-3, IL=1e-3), default_state='SH', name='sis')
        return ss.Sim(sis, unit='day', stop=30, total_pop=10_000, verbose=0)
    assert np.isclose(make().run().results.sis.IH[-1], 1003.3, atol=1) # Keeling and Rohani's program 3.1: states are people unless declared people=False (they were guessed to be counts, so 9,000 infected with only a note)
    return


@sc.timer()
def test_junction():
    sc.heading('Testing that a state left only by branches with no wait is passed through at once')
    for solver in ['ode', 'abm']: # As agents, people were held in J for a step
        S, J, A, B = ss.states('S J A B')
        d = ss.Disease(go=S >> J @ ss.perday(0.1), a=ss.Transition(J, A, p=0.3), b=ss.Transition(J, B, p=0.7), name='d')
        res = ss.Sim(d, unit='day', stop=5, total_pop=1000, n_agents=1000, solver=solver, verbose=0).run().results.d
        assert res.J.sum() == 0 and res.A[1] > 0
    return


@sc.timer()
def test_immediate():
    sc.heading('Testing dur=0: a move with no wait, on entry or as soon as a condition holds, under every solver')
    size = lambda solver: dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
    for solver in ['ode', 'binomial', 'gillespie', 'abm']: # A junction holds nobody: a split, and a single destination
        S, P, A, B = ss.states('S P A B')
        d = ss.Disease(go=ss.Transition(S, P, rate=ss.perday(0.1)), triage=ss.Transition(P, {A: 0.8, B: None}, dur=0), name='d')
        res = ss.Sim(d, unit='day', stop=10, solver=solver, **size(solver), **kw).run().results.d
        assert res.P.max() == 0 and np.isclose(res.A[-1]/(res.A[-1] + res.B[-1]), 0.8, atol=0.03)
        S, P, A = ss.states('S P A')
        d = ss.Disease(go=ss.Transition(S, P, rate=ss.perday(0.1)), onward=ss.Transition(P, A, dur=0), name='d')
        res = ss.Sim(d, unit='day', stop=10, solver=solver, **size(solver), **kw).run().results.d
        assert res.P.max() == 0 and res.onward.sum() == res.go.sum()

    # With where=, as soon as the condition holds: everyone moved to level b recovers at once, under compartments too
    out = {}
    for solver in ['ode', 'abm']:
        I, R = ss.states('I R')
        d = ss.Disease(I, R, drop=ss.Transition(I, R, dur=0, who=ss.conditions.Cond('dim', 'level') == 'b'), default_state='I', name='d')
        d.stratify(level={'a': 1, 'b': 0}, switch=ss.transfer(dict(level='b'), rate=ss.perday(0.1)))
        out[solver] = ss.Sim(d, unit='day', stop=20, solver=solver, **size(solver), **kw).run().results.d.I[-1]/10_000
    assert np.isclose(out['ode'], np.exp(-2), rtol=0.01) and np.isclose(out['abm'], out['ode'], atol=0.02)

    # As an ODE, I -> X -> R with dur=0 out of X takes about as long as I -> R (before, X >> R @ ss.perday(1e6) took 48 s for 30 days)
    def make(through):
        S, I, X, R = ss.states('S I X R')
        trs = dict(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, X if through else R, dur=10))
        trs.update(through=ss.Transition(X, R, dur=0)) if through else None
        return ss.Disease(**trs, init_prev=0.01, name='sir')
    times = {}
    for through in [False, True, False, True]:
        T = sc.timer()
        res = ss.Sim(make(through), unit='day', dur=ss.days(3650), **kw).run().results.sir
        times.setdefault(through, []).append(T.toc(output=True))
    assert res.X.max() == 0 and min(times[True]) < 2*min(times[False]) + 0.05, f'dur=0 took {min(times[True]):.3f} s, against {min(times[False]):.3f} s without it'

    # A split with p= and no rate or duration still raises, saying dur=0
    with pytest.raises(ValueError, match='dur=0'):
        S, P, A, B = ss.states('S P A B')
        ss.Disease(go=ss.Transition(S, P, rate=0.1), triage=ss.Transition(P, {A: 0.8, B: None}), name='d')
    return res


@sc.timer()
def test_level_change_dwell():
    sc.heading('Testing that a dwell drawn at one level follows a person who moves to another, where the transition runs at a different speed')
    out = {}
    for solver in ['ode', 'abm']: # Recovery is ten times slower in b, and people move there at once; as agents they kept the dwell drawn in a (KI6: 0.37 still infected at day 20 against 0.83)
        I, R = ss.states('I R')
        d = ss.Disease(rec=I >> R @ ss.days(20), default_state='I', name='d')
        d.stratify(level={'a': 1, 'b': 0}, switch=ss.transfer(dict(level='b'), rate=ss.perday(0.5)))
        out[solver] = ss.Sim(d, connectors=ss.modify(d.rec, multiply=ss.by(level=[1, 0.1])), unit='day', dt=0.25, stop=20, total_pop=20_000, n_agents=20_000, solver=solver, verbose=0).run().results.d.I[-1]/20_000
    assert np.isclose(out['abm'], out['ode'], atol=0.03)
    return


def test_dwell_expressions():
    """ A dwell time that is an expression of named parameters keeps their names: two of them clashed with the name made for the slot (DuplicateDeclaration), and ss.par(dist)*1 raised AttributeError """
    sc.heading('Testing dwell times as expressions of parameters')
    S, I, J = ss.states('S I J')
    k = ss.attr(ss.uniform(0.5, 1.5))
    dur_rec = ss.par(ss.lognormal(ss.years(1), ss.years(1)), name='dur_rec')
    dur_go = ss.par(ss.lognormal(ss.years(1), ss.years(1)), name='dur_go')
    d = ss.Disease(S, I, J, infection=S >> I @ ss.peryear(1)*I, rec=ss.Transition(I, J, dur=dur_rec*k), go=ss.Transition(J, S, dur=dur_go*k), k=k, init_prev=0.1, name='d')
    sim = ss.Sim(d, n_agents=1000, start=2000, stop=2003, verbose=0, pars=dict(dur_rec=ss.lognormal(ss.years(2), ss.years(1)))).run()
    assert 'dur_rec' in sim.diseases.d.pars and 'dur_go' in sim.diseases.d.pars and sim.results.d.rec.sum() > 0
    A, B = ss.states('A B')
    one = ss.Disease(A, B, ab=ss.Transition(A, B, dur=ss.par(ss.lognormal(ss.years(1), ss.years(1)), name='dur_ab')*1), init=dict(B=0), default_state='A', name='one')
    assert ss.Sim(one, n_agents=500, start=2000, stop=2002, verbose=0).run().results.one.ab.sum() > 0

    # A distribution times a factor scales each draw, so keeps its shape: times an attribute equal to 1, it is the distribution itself (it was an exponential of the same mean)
    def still_in(mode, factor=1.0, years=(1, 5, 20, 50)):
        S, I, R = ss.states('S I R')
        k = ss.attr(factor)
        d0 = ss.lognormal(ss.years(5), ss.years(20))
        dur = dict(plain=d0, attr=d0*k, par=ss.par(ss.lognormal(ss.years(5), ss.years(20)), name='dd')*k)[mode]
        d = ss.Disease(S, I, R, rec=ss.Transition(I, R, dur=dur), k=k, init_n=dict(I=20_000), name='d')
        n = ss.Sim(d, n_agents=20_000, start=2000, stop=2060, dt=0.25, verbose=0).run().results.d.I
        return np.array([n[int(y/0.25)]/20_000 for y in years])
    plain = still_in('plain')
    for mode in ['attr', 'par']:
        assert np.allclose(still_in(mode), plain, atol=0.02), f'{mode}: {still_in(mode)} against {plain}'
    doubled = still_in('attr', factor=2.0, years=(2, 10, 40))
    assert np.allclose(doubled, plain[:3], atol=0.02), f'Twice as long, so at twice the times: {doubled} against {plain[:3]}'
    return sim


@sc.timer()
def test_schedule():
    """ Exits are scheduled at entry: p= is read then, and nobody in a state has no scheduled exit """
    sc.heading('Testing exits scheduled at entry')

    # A split read at entry: everyone infected in the first five steps is severe, including those infected in the fifth
    early = lambda sim: 1.0*(sim.ti < 5) # The first five steps are 0 to 4
    def check(sim): # An HIV-like hook after the step: the share of the acute dwell elapsed needs the scheduled exit, for those who entered this step too
        d = sim.diseases.d
        for name in ['A', 'R']: # The states with a scheduled exit
            u = d[name].uids
            assert not np.isnan(d[f'ti_exit_{name}'][u]).any(), f'{len(u)} people in {name} have no scheduled exit at step {sim.ti}'
    S, A, H, R = ss.states('S A H R')
    d = ss.Disease(
        infection = ss.Transition(S, A, rate=0.4 * A),
        severe    = ss.Transition(A, H, p=early, dur=3),
        mild      = ss.Transition(A, R, p=lambda sim: 1 - early(sim), dur=3),
        relapse   = ss.Transition(R, A, dur=20), # Re-entered in the state-change row, so an exit is needed the same step
        check     = check,
        init_prev = 0.01, name='d',
    )
    sim = ss.Sim(d, n_agents=5000, unit='day', stop=40, **kw).run()
    res = sim.results.d
    assert res.severe.sum() == res.A[0] + res.infection[:5].sum() + res.relapse[:5].sum() # Seeds whose dwell rounds to no steps leave at the start, before the first time point's count, so are not a flow

    # Where= on a branch that is closed to everyone at entry: they wait until it opens (FPsim)
    A, B = ss.states('A B')
    clock = ss.Disease(A, B, tick=A >> B @ ss.constant(ss.months(3)), default_state='A', name='clock') # Everyone starts in the first state
    X, Y, Z = ss.states('X Y Z')
    dec = ss.Disease(X, Y, Z, to_y=ss.Transition(X, Y, p=0.5, rate=ss.permonth(30), who=clock.B),
                     to_z=ss.Transition(X, Z, p=0.5, rate=ss.permonth(30), who=clock.B), default_state='X', name='dec')
    sim = ss.Sim([clock, dec], n_agents=1000, start=2000, stop=2001, dt=ss.months(1), unit='year', **kw).run()
    assert sim.results.dec.X[:4].min() == 1000 and sim.results.dec.X[5:].max() == 0

    # A rate of 0 when the dwell was drawn (where= on a timed transition), raised later
    A, B, X, Y = ss.states('A B X Y')
    clock = ss.Disease(A, B, tick=A >> B @ ss.constant(ss.months(3)), default_state='A', name='clock') # Everyone starts in the first state
    dec = ss.Disease(X, Y, to_y=ss.Transition(X, Y, dur=ss.months(1), who=clock.B), default_state='X', name='dec')
    sim = ss.Sim([clock, dec], n_agents=1000, start=2000, stop=2002, dt=ss.months(1), unit='year', **kw).run()
    assert sim.results.dec.X[:4].min() == 1000 and sim.results.dec.X[-1] < 10
    return sim


@sc.timer()
def test_split_timing():
    """ Separate p= transitions out of a state split on entry; one transition with several destinations splits when it is taken """
    sc.heading('Testing when a split is decided')
    late = lambda sim: 1.0*(sim.ti >= 10) # Severe only from the step that starts at 10
    def make(shape):
        I, H, R = ss.states('I H R')
        if shape == 'entry': # Branch, then wait
            trs = dict(severe=ss.Transition(I, H, p=late, dur=20), mild=ss.Transition(I, R, p=lambda sim: 1 - late(sim), dur=20))
        else: # Wait, then branch
            trs = dict(course=ss.Transition(I, {H: late, R: None}, dur=20))
        return ss.Disease(I, H, R, **trs, default_state='I', name='d') # Everyone starts in I
    for solver in ['ode', 'abm']:
        size = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
        entry = ss.Sim(make('entry'), solver=solver, stop=100, **size, **kw).run().results.d
        exit = ss.Sim(make('exit'), solver=solver, stop=100, **size, **kw).run().results.d
        assert entry.H[-1] == 0 # Everyone started before step 10, so was split then
        assert np.isclose(exit.H[-1]/10_000, np.exp(-10/20), atol=0.03) # Those still in I at step 10
    return exit


@sc.timer()
def test_infection_step():
    """ As agents, the newly infected have half a step of the rates out of an infectious state at their first draw, so R0 holds at a coarse step """
    sc.heading('Testing the final size of an SIR as agents at a coarse step')
    S, I, R = ss.states(3)
    sir = ss.Disease(S, I, R, infection=S >> I @ ss.perday(0.25)*I, recovery=I >> R @ ss.perday(1/8), init_prev=0.001, name='sir') # R0 = 2
    res = ss.Sim(sir, solver='abm', n_agents=20_000, unit='day', stop=250, dt=ss.days(4), rand_seed=1, verbose=0).run().results.sir
    attack = 1 - res.S[-1]/20_000
    assert np.isclose(attack, 0.797, atol=0.02), f'The attack rate at a step of half the infectious period is {attack:n}, not 0.797' # 0.61 with a whole step at the first draw, as if R0 were 1.5
    return attack


@sc.timer()
def test_wholestep():
    """ With solver=ss.solvers.abm(step='wholestep'), rates act over whole steps only, on those in the state at the start of the step: those who arrive during a step face them from the next; and a module's own way of running as agents """
    sc.heading('Testing rates over whole steps, and a module\'s own step')
    def make(cls=ss.Disease, **kwargs):
        A, B, C = ss.states('A B C')
        return cls(A, B, C, tick=A >> B @ ss.constant(ss.days(3)), leave=B >> C @ ss.perday(0.5), default_state='A', name='d', **kwargs) # Everyone reaches B in step 3
    res = {}
    for step in ['midstep', 'wholestep']:
        sim = ss.Sim(make(), solver=ss.solvers.abm(step=step), n_agents=10_000, unit='day', stop=10, verbose=0).run()
        eng = sim.diseases.d.engine
        assert isinstance(eng, ss.solvers.Compiled) and not eng.uncompiled
        assert eng.wholestep == (step == 'wholestep') and sim.solver.step == step
        res[step] = sim.results.d
    assert res['midstep'].tick[3] == 10_000
    assert np.isclose(res['midstep'].leave[3]/10_000, 1 - np.exp(-0.25), atol=0.02) and res['wholestep'].leave[3] == 0 # Half a step of the rate in the step they arrive, or none
    p = 1 - np.exp(-0.5)
    assert np.isclose(res['wholestep'].leave[4]/10_000, p, atol=0.02) # Then a whole step's chance each step
    assert np.isclose(res['wholestep'].B[6]/10_000, (1 - p)**2, atol=0.02)

    # A module's own way of running as agents, which the sim's does not override
    sim = ss.Sim(make(solver=ss.solvers.abm(step='wholestep')), n_agents=2000, unit='day', stop=10, verbose=0).run()
    assert sim.diseases.d.engine.wholestep and sim.results.d.leave[3] == 0
    with pytest.raises(ValueError):
        ss.Disease(solver='ode') # A module's own solver is a way of running as agents
    wholestep = lambda sim_solver, **kw: ss.Sim(make(**kw), solver=sim_solver, n_agents=500, unit='day', stop=5, verbose=0).init().diseases.d.engine.wholestep
    assert not wholestep('abm') and wholestep(ss.solvers.abm(step='wholestep'))
    assert wholestep('abm', solver=ss.solvers.abm(step='wholestep')) and not wholestep(ss.solvers.abm(step='wholestep'), solver='abm') # Its own overrides the sim's
    assert ss.solvers.agent_solver(ss.Sim(make(solver=ss.solvers.abm(step='wholestep')), n_agents=500, verbose=0).init().diseases.d, ss.Sim(n_agents=10), why=True) == ('wholestep', 'own')

    # The coarse-step SIR of test_infection_step(): the newly infected have a whole step of recovery at their first draw, as written as code with ss.bernoulli()
    S, I, R = ss.states(3)
    sir = ss.Disease(S, I, R, infection=S >> I @ ss.perday(0.25)*I, recovery=I >> R @ ss.perday(1/8), init_prev=0.001, name='sir')
    r = ss.Sim(sir, solver=ss.solvers.abm(step='wholestep'), n_agents=20_000, unit='day', stop=250, dt=ss.days(4), rand_seed=1, verbose=0).run().results.sir
    attack = 1 - r.S[-1]/20_000
    assert 0.5 < attack < 0.75, f'The attack rate is {attack:n}, not about 0.6' # 0.797 by midsteps, with half a step at the first draw
    return res


@sc.timer()
def test_compiled_draws():
    """ The compiled moves of the agent engine make the draws its own moves make, so a sim gives the same numbers whichever takes its moves (e.g. with an intervention the compiled moves do not take, and without it) """
    sc.heading('Testing that the compiled moves draw the numbers of the agent engine\'s own moves')
    def make(seed):
        S, E, I, H, R, F = ss.states('S E I H R F')
        infection = S >> E @ ss.perday(0.3)*(I + H)
        d = ss.Disease(S, E, I, H, R, F, infection=infection,
            onset   = ss.Transition(E, {I: 0.8, R: None}, dur=ss.lognormal(ss.days(3), ss.days(1))), # A split after a dwell time
            severe  = ss.Transition(I, H, p=0.2, dur=ss.weibull(shape=2, scale=ss.days(4))), # Branches chosen on entry
            mild    = ss.Transition(I, {R: 0.9, ss.stay: None}, dur=ss.uniform(ss.days(3), ss.days(9))),
            flag    = ss.Transition(E, F, when=ss.on(infection, delay=ss.days(2))), # A time after an event
            unflag  = ss.Transition(F, {I: 0.5, R: None}, dur=0), # Taken on in the same step
            release = ss.Transition(H, R, when=H.time_in > ss.days(5)), # A condition
            die     = H >> ss.dead @ ss.perday(0.03), # A competing rate beside a dwell time
            wane    = ss.Transition(R, S, dur=ss.normal(ss.by(sex=dict(female=ss.days(25), male=ss.days(35))), ss.days(5))), # Drawn by the engine after the call
            init_prev=0.02, name='d')
        return ss.Sim(d, networks=ss.RandomNet(n_contacts=4, method='safe'), n_agents=1000, unit='day', stop=60, rand_seed=seed, **kw)
    for seed in [0, 1]:
        sims = [make(seed).run()]
        with mock.patch.object(ss.solvers.agents_compiled, 'compiled_reasons', lambda engine: ['off']): # The agent engine's own moves
            sims.append(make(seed).run())
        compiled, own = [sim.diseases.d for sim in sims]
        assert not compiled.engine.uncompiled and own.engine.uncompiled == ['off']
        assert ss.debugtools.check_sims_match(*sims)
        for k,v in compiled.__dict__.items():
            if isinstance(v, ss.Arr):
                assert np.array_equal(v.raw, own[k].raw, equal_nan=True), f'{k} differs between the compiled moves and the agent engine\'s own'
    return sims


@sc.timer()
def test_lifelong():
    """ A function marked with ss.lifelong() is worked out once for each person, and gives the same run as unmarked """
    sc.heading('Testing values fixed for life')
    female = lambda sim: np.asarray(sim.people.female.raw, dtype=bool)
    share = lambda sim, uids: np.where(sim.people.female[uids], 0.8, 0.2)
    def make(mark):
        f = ss.lifelong if mark else (lambda func: func)
        S, I, H, R = ss.states('S I H R')
        return ss.Disease(S, I, H, R,
            infection = ss.Transition(S, I, rate=0.3*I),
            course    = ss.Transition(I, {H: f(share), R: None}, dur=ss.days(8)),
            recover   = ss.Transition(H, R, dur=ss.days(10), who=f(female)), # Men stay in H
            init_prev = 0.02, name='d')
    sims = [ss.Sim(make(mark), n_agents=5000, unit='day', stop=60, demographics=[ss.Births(birth_rate=ss.peryear(0.03))], rand_seed=3, verbose=0) for mark in [False, True]]
    assert same(*sims)
    eng = sims[1].diseases.d.engine
    assert not eng.krescaled and dict(zip([k for k,_ in eng.kedges], [kind for kind,_ in eng.kedge_src]))[('dest', 'course')] == 'table'
    plain = sims[0].diseases.d.engine
    assert plain.krescaled and dict(zip([k for k,_ in plain.kedges], [kind for kind,_ in plain.kedge_src]))[('dest', 'course')] == 'vary'
    key = ('dest', 'course')
    table, done = eng.lifelong[key], eng.lifelong_used[key] # Worked out again for those born into the room the arrays had for them, once read for them
    assert done > 5000 and np.array_equal(table[:done, 1], np.where(sims[1].people.female.raw[:done], 0.8, 0.2)) # The cumulative shares: 0, H's, 1
    m = ss.lifelong(share)
    assert m is not share and m(sims[1], ss.uids([0, 1])).shape == (2,) and not hasattr(share, 'lifelong') # A marked copy
    return sims


@sc.timer()
def test_stays():
    sc.heading('Testing the time people spent in a state (E.stays): the declared mean, where the difference of the times of entering is a step long')
    S, E, I, R = ss.states('S E I R')
    seir = ss.Disease(S >> E @ 1.9*I, E >> I @ ss.days(10), I >> R @ ss.days(8), init_n=dict(I=10), name='seir')
    sim = ss.Sim(seir, n_agents=5000, dur=ss.days(200), rand_seed=1, verbose=0).run()
    m = sim.diseases.seir
    stays = m.E.stays
    assert isinstance(stays, ss.days) and len(stays) > 4000 and np.isclose(np.mean(stays).value, 10, atol=0.5), np.mean(stays) # The seeds, who never entered E, are not counted
    done = (m.R | m.I).uids[np.isfinite(m.ti_E[(m.R | m.I).uids])]
    assert np.isclose(np.mean(m.ti_I[done] - m.ti_E[done]), np.mean(stays).value + 1, atol=0.1) # As read, a step long (see ss.TimeArr)
    assert np.isclose(np.mean(m.I.stays).value, 8, atol=0.5), np.mean(m.I.stays) # Without keep_times: the time of leaving is kept for every state whose time of entering is
    with pytest.raises(AttributeError, match='ti_R is not kept'): # But not the time of entering where it leads
        m.ti_R.values

    # Going back to a state it led to (a relapse) does not change when people left
    E, I, R = ss.states('E I R')
    tb = ss.Disease(E >> I @ ss.days(5), I >> R @ ss.days(5), R >> I @ ss.days(5), init_n=dict(E=1000), name='tb')
    m = ss.Sim(tb, n_agents=1000, dur=ss.days(60), verbose=0).run().diseases.tb
    assert len(m.E.stays) == 1000 and np.isclose(np.mean(m.E.stays).value, 5, atol=0.5), np.mean(m.E.stays) # Not from the last time they entered I
    return stays


@sc.timer()
def test_stay_beside_dwell():
    sc.heading('Testing a minimum stay (after=) on one exit beside the dwell time of another, as compartments: the dwell runs from entry, as for agents')
    def make(dur, after=True, rate=0.0, solver='ode'):
        A, E, I, D = ss.states('A E I D')
        die = dict(die=ss.Transition(E, D, rate=ss.perday(rate), min_stay=ss.days(2) if after else None))
        d = ss.Disease(A, E, I, D, enter=A >> E @ ss.perday(0.3), onset=ss.Transition(E, I, dur=dur), **die, init=dict(A=1.0), default_state='A', name='d')
        return ss.Sim(d, unit='day', stop=20, total_pop=1e5, solver=solver, verbose=0).run()
    I = lambda sim: sim.results.d.I.values
    for dur in [ss.normal(ss.days(5), ss.days(1)), ss.choice(a=ss.days([3, 4, 5]), p=[0.2, 0.5, 0.3])]: # As stages in each slot of the stay, and as a conveyor of whole days
        assert np.allclose(I(make(dur)), I(make(dur, after=False)), rtol=1e-6, atol=0.01), 'A stay on an exit nobody takes does not delay the onset' # To the accuracy of the ODE: the stages are shifted along the conveyor at the end of each step
    E, X, D = ss.states('E X D') # A hazard acts on the people in every slot of another exit's stay, the last too
    d = ss.Disease(E, X, D, die=ss.Transition(E, D, rate=ss.perday(0.2)), other=ss.Transition(E, X, rate=ss.perday(0.0), min_stay=ss.days(2)), init=dict(E=1.0), default_state='E', name='d')
    sim = ss.Sim(d, unit='day', stop=6, total_pop=1e5, solver='ode', verbose=0).run()
    assert np.allclose(sim.results.d.E.values/1e5, np.exp(-0.2*np.arange(7)), rtol=1e-4), sim.results.d.E.values # It was 0.61 at day 3, not 0.55, when the hazard left out the last slot
    return


@sc.timer()
def test_held_rate():
    sc.heading('Testing a rate held in a parameter (ss.par(beta*I)): it is that rate, under every solver')
    def make(held, solver='ode', **kwargs):
        S, I, R = ss.states('S I R')
        rate = ss.perday(0.3)*I
        sir = ss.Disease(infection=S >> I @ (ss.par(rate, name='force') if held else rate), recovery=I >> R @ ss.days(8), init=dict(I=0.01), name='sir')
        return ss.Sim(sir, unit='day', stop=60, total_pop=10_000, solver=solver, verbose=0, **kwargs)
    for solver,kwargs in [('ode', {}), ('abm', dict(n_agents=2000, networks=ss.RandomNet(n_contacts=4)))]: # Worked out within the step, and as a transmission over a network
        held, plain = make(True, solver, **kwargs).run(), make(False, solver, **kwargs).run()
        assert np.array_equal(held.results.sir.I.values, plain.results.sir.I.values), solver
    assert held.diseases.sir.graph.transitions.infection.contact and np.isclose(make(True).init().diseases.sir.R0(), 2.4)
    with pytest.raises(ValueError, match='cannot be fitted'):
        S, I, R = ss.states('S I R')
        ss.Sim(ss.Disease(infection=S >> I @ ss.par(ss.perday(0.3)*I, bounds=[0.1, 1]), recovery=I >> R @ ss.days(8)), verbose=0).init()
    return


@sc.timer()
def test_cascade():
    """ Moves with no wait happen in the step that caused them, under every solver """
    sc.heading('Testing the same-step cascade')
    def make():
        S, X, E, P, A, Q, R = ss.states('S X E P A Q R')
        return ss.Disease(S, X, E, P, A, Q, R,
            infection = S >> E @ 0.4*(P + A + Q),
            onset     = ss.Transition(S, X, rate=ss.perday(0.002)), # Into an immediate split in the state-change row, not by transmission
            entry     = ss.Transition(X, E, dur=2),
            split_p   = ss.Transition(E, P, p=0.3), # No driver: split on entry, at once
            split_a   = ss.Transition(E, A, p=0.7),
            onward    = ss.Transition(P, Q, p=1.0), # And on again, in the same step
            rec_q     = Q >> R @ ss.days(5),
            rec_a     = A >> R @ ss.days(5),
            init_n=dict(E=50), name='d')
    for solver in ['ode', 'abm']:
        size = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
        res = ss.Sim(make(), solver=solver, unit='day', stop=60, **size, **kw).run().results.d
        for name in ['E', 'P']: # Nobody is ever counted in a state with no wait, including at the start
            assert res[name].max() == 0, f'{solver}: {res[name].max()} people in {name}'
        into = res.infection[1:] + res.entry[1:]
        split = res.split_p + res.split_a
        assert np.allclose(split[1:], into) # In the same step as whatever brought them
        assert np.isclose(res.onward.sum(), 0.3*split.sum(), rtol=0.1)

    # As agents, seeds that move on at the start are events seen in the first step (on=), and are counted in its flows
    for junction in [True, False]: # Before the start (E is a junction), or in the first step
        S, E, C = ss.states('S E C')
        onset = ss.Transition(E, C, p=1.0) if junction else ss.Transition(E, C, dur=0)
        ng = ss.Disease(infection=ss.Transition(S, E, rate=0.0 * E), onset=onset, recovery=ss.Transition(C, S, dur=ss.days(300)), init_prev=0.1, name='ng')
        seek = ss.treat(ss.Treatment(cure=0.0), when=ng.C.enter, name='seek')
        res = ss.Sim(ng, interventions=seek, n_agents=1000, unit='day', stop=10, **kw).run().results
        assert res.seek.delivered.sum() == 100, 'Each seed is news to the deliveries once'
        assert np.array_equal(res.ng.onset.values[:-1], res.seek.delivered.values[1:]), 'The flow (at the start) should count the moves the deliveries saw, in the first step'

    # A dwell time competing with a hazard, for people who enter during the run (the hazard missed a step per entrant)
    S, A, B, C = ss.states('S A B C')
    d = ss.Disease(S, A, B, C, sa=S >> A @ ss.peryear(0.2), ab=ss.Transition(A, B, dur=ss.months(3)), ac=A >> C @ ss.permonth(0.04), name='d')
    res = ss.Sim(d, n_agents=50_000, start=2000, stop=2010, dt=ss.months(1), unit='year', aging=False, **kw).run().results.d
    share = res.ac.sum()/(res.ac.sum() + res.ab.sum())
    assert np.isclose(share, 0.1075, atol=0.005), f'{share:.3f} went to C; with whole steps of a month, 0.1075 should' # The mean over seeds (sd 0.002); 0.107 in continuous time

    # A loop of moves with no wait never settles
    for solver in ['ode', 'abm']:
        S, A, B = ss.states('S A B')
        loop = ss.Disease(S, A, B, sa=S >> A @ 0.1, ab=ss.Transition(A, B, p=1.0), ba=ss.Transition(B, A, p=1.0), name='loop')
        size = dict(n_agents=1000) if solver == 'abm' else {}
        with pytest.raises(ValueError):
            ss.Sim(loop, solver=solver, stop=5, **size, **kw).run()
    return res


def test_cascade_binomial():
    """ Under the stochastic compartmental solvers, the moves through transitions with no wait are counted as the people who passed through them """
    sc.heading('Testing the same-step cascade')
    def make():
        S, X, E, P, A, Q, R = ss.states('S X E P A Q R')
        return ss.Disease(S, X, E, P, A, Q, R,
            infection = S >> E @ 0.4*(P + A + Q),
            onset     = ss.Transition(S, X, rate=ss.perday(0.002)), # Into an immediate split in the state-change row, not by transmission
            entry     = ss.Transition(X, E, dur=2),
            split_p   = ss.Transition(E, P, p=0.3), # No driver: split on entry, at once
            split_a   = ss.Transition(E, A, p=0.7),
            onward    = ss.Transition(P, Q, p=1.0), # And on again, in the same step
            rec_q     = Q >> R @ ss.days(5),
            rec_a     = A >> R @ ss.days(5),
            init_n=dict(E=50), name='d')
    for solver in ['binomial', 'gillespie']: # Each person once, through one branch, and on from P only (they were drawn as exclusive, so onward took from the others)
        res = ss.Sim(make(), solver=solver, unit='day', stop=60, total_pop=5000, **kw).run().results.d
        into = res.infection[1:] + res.entry[1:]
        assert np.isclose((res.split_p + res.split_a).sum(), into.sum(), rtol=0.03), solver
        assert res.onward.sum() == res.split_p.sum() and np.isclose(res.split_p.sum(), 0.3*into.sum(), rtol=0.1), solver
    return res


@sc.timer()
def test_change_in_function():
    sc.heading('Testing that a function that reads a parameter sees a change to it')
    def initiation(sim, uids): # A logistic rate with an intercept: arithmetic on it gives its value now (float() gives the value it holds, without the change)
        b0 = sim.diseases.meth.pars.b0
        return ss.peryear(1/(1 + np.exp(-b0))*np.ones(len(uids)))
    def meth_model(rate):
        none, on = ss.states('none on')
        return ss.Disease(start=ss.Transition(none, on, rate=rate), stop=ss.Transition(on, none, rate=ss.peryear(0.5)), init=dict(none=1.0), b0=ss.par(-1.0), default_state='none', name='meth')
    starts = []
    for change in [False, True]:
        m = meth_model(initiation)
        intv = ss.modify('meth.pars.b0', set=-20.0, when=ss.schedule(start=2005)) if change else None
        sim = ss.Sim(m, interventions=intv, start=2000, stop=2010, dt=1/12, n_agents=2000, verbose=0).run()
        starts.append(sim.results.meth.start[61:].sum())
    assert starts[0] > 100 and starts[1] == 0
    return


@sc.timer()
def test_value():
    sc.heading('Testing that a parameter has one value for code: arithmetic on self.pars.x, declared or written as code')
    sir = ss.SIR(beta=0.3)
    p = sir.pars.beta
    assert all(isinstance(x - p, ss.expressions.BinOp) for x in [1., np.array(1.), np.ones(2)]), 'Before the sim is initialized, arithmetic on a parameter is part of the model'
    p = ss.Sim(sir, n_agents=200, verbose=0).init().diseases.sir.pars.beta
    for x in [1., np.array(1.), np.ones(2)]: # After, it is a value, whatever the shape of the other operand
        assert np.all(x - p == 0.7) and not isinstance(x - p, ss.expressions.Expr)

    class Mosquitoes(ss.Module): # Read from code: values that vary as their value now, numbers as numbers in the sim's unit; a typed value, or one that varies of typed values, typed
        def __init__(self):
            super().__init__()
            self.define_pars(bites=ss.perday(0.3), f=ss.by(sex=[0.6, 0.8]), em=ss.timeseries({'2000-01-01':1/365, '2002-01-01':2/365}),
                             death=ss.peryear(2), death_ts=ss.timeseries({'2000-01-01':ss.peryear(1), '2002-01-01':ss.peryear(2)}))
            self.probs = []
        def step(self):
            sir = self.sim.diseases.sir
            self.probs.append([self.pars.death.to_prob(), self.pars.death_ts.to_prob()]) # The same code, whichever form the rate is given in
            self.read = [self.pars.bites, 1*self.pars.f, np.exp(self.pars.em) - 1, float(self.pars.em), self.pars.em > 0, 2*self.pars.death_ts, 1*sir.pars.beta, float(sir.pars.beta)]
    sim = ss.Sim(ss.SIR(beta=0.3), modules=Mosquitoes(), interventions=ss.modify('sir.pars.beta', set=0.1, when=ss.schedule(start='2001-01-01')), start='2000-01-01', stop='2003-01-01', unit='day', dt=ss.years(1), n_agents=200, verbose=0).run()
    mosq = [m for m in sim.modules if isinstance(m, Mosquitoes)][0]
    bites, f, em, em_float, positive, death2, beta, raw = mosq.read
    assert isinstance(bites, ss.perday) and np.allclose(f, [0.6, 0.8]) and np.isclose(em, np.expm1(2/365)) and np.isclose(em_float, 2/365) and positive
    probs = np.array(mosq.probs)
    assert np.allclose(probs[:,0], -np.expm1(-2)) and np.allclose(probs[:,1], -np.expm1(-np.array([1, 1, 2, 2]))) and np.isclose(*probs[-1]), 'A typed rate and a time series of them convert alike' # A step from each year, 2003 included
    assert isinstance(death2, ss.peryear) and death2.value == 4, 'Arithmetic on a time series of typed values is typed arithmetic'
    assert np.isclose(beta, 0.1) and raw == 0.3, 'Arithmetic applies the change (to 0.1 per day); float() is the value the parameter holds'
    assert isinstance(ss.timeseries({2000:1, 2001:2})*2, ss.expressions.Expr), 'Outside a sim, it is an expression'

    # A distribution has no single value, and the sim's settings are frozen once it is initialized
    S, I, R = ss.states(3)
    sim = ss.Sim(ss.Disease(S >> I @ 0.3*I, ss.Transition(I, R, dur=ss.lognormal(8, 2)), init=dict(I=0.01), name='d'), unit='day', stop=10, verbose=0).init()
    with pytest.raises(TypeError):
        1/sim.diseases.d.pars.dur_I
    with pytest.raises(RuntimeError, match=r'sim.new\(stop='):
        sim.pars.stop = 30
    sim.label = 'renamed' # Not a setting of the run
    assert sim.new(stop=30).run().t.tvec[-1] == 30
    return


@sc.timer()
def test_given_dwell():
    sc.heading('Testing a branch probability and a rate that depend on the time in the state (P.time_in), as agents only')
    n = 20_000
    def make(solver, p=None, rate=None, dur=ss.expon(ss.years(1))):
        P, C, R = ss.states('P C R')
        exits = dict(course=ss.Transition(P, {C: p(P), R: None}, dur=dur)) if p is not None else dict(clear=P >> R @ rate(P), death=P >> ss.dead @ ss.peryear(0.1))
        return ss.Sim(ss.Disease(P, C, R, **exits, default_state='P', name='d'), n_agents=n, total_pop=n, dur=ss.years(15), dt=ss.years(0.1), solver=solver, verbose=0)

    # A known answer: with an exponential dwell of mean 1, P(C) = 1 - exp(-dwell) sends half to C
    share = make('abm', p=lambda P: 1 - np.exp(-P.time_in)).run().results.d.C[-1]/n
    assert np.isclose(share, 0.5, atol=3*np.sqrt(0.25/n)), share
    for solver in ['ode', 'binomial']: # A compartmental solver refuses, and auto runs agents
        with pytest.raises(ss.CapabilityError, match='P.time_in'):
            make(solver, p=lambda P: 1 - np.exp(-P.time_in)).init()
    assert isinstance(make('auto', p=lambda P: 1 - np.exp(-P.time_in)).init().solver, ss.solvers.abm)

    # As agents, the same as a function of the sim reading the time in the state
    func = lambda sim, uids: 1 - np.exp(-sim.diseases.d.P.time_in[uids])
    a = make('abm', p=lambda P: 1 - np.exp(-P.time_in)).run().results.d.C
    b = make('abm', p=lambda P: func).run().results.d.C
    assert np.array_equal(a, b)

    # A hazard that grows with the time in the state, against a death rate: the share cleared is 0.841
    cleared = make('abm', rate=lambda P: ss.peryear(0.5)*P.time_in).run().results.d.R[-1]/n
    assert np.isclose(cleared, 0.841, atol=0.01), cleared

    # describe() says so, and a branch with p= (decided on entry) cannot read it
    sim = make('abm', p=lambda P: 1 - np.exp(-P.time_in), dur=ss.lognormal(ss.years(3), ss.years(9))).init()
    assert 'decided as people leave' in sim.describe()
    P, C, R = ss.states('P C R')
    with pytest.raises(ss.CapabilityError):
        ss.Disease(P, C, R, a=ss.Transition(P, C, dur=ss.years(3), p=P.time_in/10), b=ss.Transition(P, R, dur=ss.years(3), p=None), init_n=dict(P=10))
    return

@sc.timer()
def test_init_prev_forms():
    sc.heading('Testing init_prev as a function and by people\'s properties: for each agent, or for each group')
    def make(**kwargs):
        S, Ic, In, R = ss.states('S Ic In R')
        return ss.Disease(infection=ss.Transition(S, {Ic: 0.3, In: 0.7}, rate=0.1*(Ic + In)), clear_c=Ic >> R @ ss.years(1), clear_n=In >> R @ ss.years(1), name='d', **kwargs)
    by_age = ss.by(dict(age=[0, 20], value=[0.0, 0.2])) # By each person's own age
    for kwargs in [dict(init_prev=by_age), dict(init_prev=lambda sim, uids: np.where(sim.people.age[uids] >= 20, 0.2, 0.0)), dict(init=dict(Ic=by_age))]:
        sim = ss.Sim(make(**kwargs), n_agents=20_000, stop=2001, **kw).init()
        uids = sim.people.auids
        old = sim.people.age.raw[uids] >= 20
        seeded = sim.diseases.d.Ic.raw[uids] | sim.diseases.d.In.raw[uids]
        assert isinstance(sim.solver, ss.solvers.abm) and seeded[~old].sum() == 0 and np.isclose(seeded[old].mean(), 0.2, atol=0.01)
        if 'init_prev' in kwargs: # Through the transmission, in its shares
            assert np.isclose(sim.diseases.d.Ic.raw[uids].sum()/seeded.sum(), 0.3, atol=0.03)
        with pytest.raises(ss.CapabilityError): # A property of each person that is not a dimension, or the agents, have no compartmental form
            ss.Sim(make(**kwargs), solver='ode', stop=2001, **kw).init()
    ode = ss.Sim(make(init_prev=lambda sim: 0.1), solver='ode', total_pop=1000, stop=2001, **kw).init() # A function of the sim alone works on groups
    assert np.isclose(ode.diseases.d.engine.count(ode.diseases.d.graph.people_states[1:3]), 100)
    by_risk = ss.Sim(make(init_prev=ss.by(risk=[0.0, 0.2])).stratify(risk=['low', 'high']), solver='ode', total_pop=1000, stop=2001, **kw).init() # By a dimension, for each group
    assert np.isclose(by_risk.diseases.d.engine.count(by_risk.diseases.d.graph.people_states[1:3]), 100)

    # A branch share by level at seeding (it crashed as agents), under both solvers
    for solver in ['abm', 'ode']:
        S, A, B = ss.states('S A B')
        d = ss.Disease(S, A, B, infection=ss.Transition(S, {A: ss.by(risk=[0.2, 0.8]), B: None}, rate=ss.peryear(0.0)*A), init_prev=0.1, name='d')
        d.stratify(risk=['lo', 'hi'])
        res = ss.Sim(d, solver=solver, total_pop=10_000, n_agents=10_000, start=2000, stop=2000.25, dt=ss.years(0.25), **kw).run().results.d
        assert np.allclose(res.A.by('risk').values[0], [100, 400])

    # Starting values set through pars after the disease is made raise, rather than being ignored (nobody was seeded)
    A, B = ss.states('A B')
    d = ss.Disease(A, B, go=A >> B @ 0.1, default_state='A', name='d')
    d.pars.init_n = dict(B=100.0)
    with pytest.raises(KeyError, match='init='):
        ss.Sim(d, n_agents=1000, start=2000, stop=2001, **kw).init()
    return sim


@sc.timer()
def test_split_null():
    sc.heading('Testing ss.dead as one destination of a split: a share die when the transition is taken, as a death would')
    def make(dst, p):
        S, I, R = ss.states('S I R')
        dst = [R if d == 'R' else d for d in dst]
        return ss.Disease(S, I, R, infection=ss.Transition(S, I, rate=0.0), death=ss.Transition(I, dict(zip(dst, p)), dur=ss.days(10)), init=dict(S=0.5, I=0.5), name='d') # Nobody is infected: those in I at the start each take it once
    out = sc.obj()
    for solver in ['abm', 'ode', 'binomial']:
        size = dict(n_agents=20_000) if solver == 'abm' else dict(total_pop=20_000)
        sim = ss.Sim(make([ss.dead, ss.stay], [0.05, None]), unit='day', stop=200, solver=solver, verbose=0, **size).run()
        r = sim.results.d
        n0 = r.I[0]
        died = r.death.values.sum()
        assert np.isclose(died/n0, 0.05, rtol=0.15), f'{solver}: {died/n0:n} of those in I died, not 0.05'
        assert np.isclose(r.I[-1]/n0, 0.95, rtol=0.01), f'{solver}: the rest should stay in I, never taking it again, but {r.I[-1]/n0:n} are left'
        if solver == 'abm':
            assert sim.results.died.values.sum() == died # Counted as deaths, as a transition to ss.dead is
        out[solver] = died/n0

        # With a state: the deaths are res.<transition>.dead
        sim = ss.Sim(make(['R', ss.dead], [0.9, 0.1]), unit='day', stop=200, solver=solver, verbose=0, **size).run()
        r = sim.results.d
        n0 = r.I[0]
        assert np.isclose(r['death.dead'].values.sum()/n0, 0.1, rtol=0.1) and np.isclose(r.R[-1]/n0, 0.9, rtol=0.02), solver
        assert np.isclose(r['death.dead'].values.sum() + r['death.R'].values.sum(), r.death.values.sum())
        assert np.isclose(r.S[-1] + r.I[-1] + r.R[-1] + r['death.dead'].values.sum(), r.S[0] + n0), f'{solver}: everyone is in a state or dead'
        if solver == 'abm':
            assert sim.results.died.values.sum() == r['death.dead'].values.sum()
    assert np.isclose(out.abm, out.ode, rtol=0.15)

    # The messages for a source or destination that is not a state or a list of states
    S, I = ss.states('S I')
    with pytest.raises(TypeError, match=r'ss.dead.*alone'):
        ss.Sim(ss.Disease(S, I, x=ss.Transition([I, ss.dead], S, rate=0.1), name='x'), n_agents=100, verbose=0).init()
    S, E, P, A = ss.states('S E P A')
    assert 'prevalence' in ss.Disease(S, E, infection=S >> E @ 0.3*E, recovery=E >> S @ ss.days(5)).graph.exprs # From the force of infection: E is infected
    assert 'E + P + A' in repr(ss.Transition(E | P | A, S, rate=0.1)) # Printed before it is built
    with pytest.raises(TypeError, match=r'\[E, P, A\].*E \+ P \+ A'):
        ss.Sim(ss.Disease(S, E, P, A, x=ss.Transition(E | P | A, S, rate=0.1), name='x'), n_agents=100, verbose=0).init()

    # A split to one destination is a plain transition, so several out of one state compete as rates rather than summing as branches
    S, I, R, F = ss.states('S I R F')
    d = ss.Disease(S >> I @ 0.3*I, a=ss.Transition(I, [F], p=[1.0], rate=0.1), b=ss.Transition(I, [R], p=[1.0], rate=0.05), name='d')
    assert d.a.p is None and len(ss.Sim(d, n_agents=100, verbose=0).init().diseases.d.graph.exits.I.hazards) == 2
    return


@sc.timer()
def test_read_each_step():
    sc.heading('Testing what a run taken whole, and one stepped, read at each step: other diseases, modifiers, and values in time')
    class Step(ss.Intervention):
        def step(self):
            if self.sim.t.ti == 14 and 'm' in self.sim.connectors: # From the step from day 14
                self.sim.connectors.m.amount = 0.1

    def sir(name='sir', rate=lambda I: 0.3*I, results=lambda I: {}):
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=ss.Transition(S, I, rate=rate(I)), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name=name, **results(I))

    pars = dict(unit='day', stop=200, total_pop=50_000, solver='ode', verbose=0)
    def pair():
        a = sir('a')
        return [a, sir('b', rate=lambda I: 0.3*I*(1 + 5*a.I.frac))]
    whole, stepped = [ss.Sim(pair(), interventions=ints, **pars).run().results.b.I for ints in [None, Step(name='step')]] # A count of another disease is not frozen
    assert np.allclose(whole, stepped) and np.isclose(np.max(whole), 25_059, rtol=1e-3)

    changed = ss.Sim(sir(), connectors=ss.modify('sir.infection', multiply=0.5, name='m'), interventions=Step(name='step'), **pars).run()
    assert np.isclose(np.max(changed.results.sir.I), 959.4, rtol=1e-3) # A coded step writing a modifier's amount

    b = ss.par(ss.seasonal(ss.perday(0.3), 0.5), name='b')
    w = ss.par(ss.seasonal(1.0, 0.8), name='w') # Only in a result
    make = lambda: sir(rate=lambda I: b*I, results=lambda I: dict(foi=b*I, wI=w*I))
    pars = sc.mergedicts(pars, dt=30, stop=720)
    whole, stepped = [ss.Sim(make(), interventions=ints, **pars).run().results.sir for ints in [None, Step(name='step')]]
    assert np.allclose(whole.foi, stepped.foi) and np.allclose(whole.wI, stepped.wI)
    assert np.allclose((np.array(whole.foi)/np.array(whole.I))[:3], [0.45, 0.43044, 0.37686], atol=1e-4) # At each time point
    assert np.allclose((np.array(whole.wI)/np.array(whole.I))[:3], [1.8, 1.69567, 1.4099], atol=1e-4)
    return


@sc.timer()
def test_sibling_event():
    sc.heading('Testing an event of a sibling in the class form')
    class Vax(ss.Disease):
        S, I, R, V = ss.states('S I R V')
        infection = ss.Transition(S, I, rate=ss.perday(0.3)*I)
        recovery = ss.Transition(I, R, dur=ss.days(5))
        vax = ss.Transition(R, V, when=recovery)
    res = ss.Sim(Vax(init_prev=0.1), n_agents=500, unit='day', stop=30, solver='abm', verbose=0).run().results.vax
    assert res.vax.sum() == res.recovery.sum() > 0 and res.R[-1] == 0
    return


def test_state_as_result():
    sc.heading('Testing a state under another name as a result: the number in it, stratified like n_<state>, with the state keeping its name')
    class D(ss.Disease):
        S, I = ss.states('S I')
        infection = S >> I @ 0.1*I
        n_infected = I
    d = D(init_prev=0.1, name='d')
    assert list(d.states.keys()) == ['S', 'I']
    sim = ss.Sim(d, n_agents=500, start=2000, stop=2005, stratify=dict(sex=['female', 'male']), verbose=0).run()
    r = sim.results.d
    assert np.array_equal(r.n_infected.values, r.I.values)
    assert np.array_equal(r.n_infected.select(sex='female'), r.I.select(sex='female'))
    return


@sc.timer()
def test_branch_ends():
    sc.heading('Testing ss.dead as a branch: by group, adjusted, and modified, with the other branches sharing what is left')
    def make(dead=0.02, by_age=False):
        S, I, R, H = ss.states('S I R H')
        dis = ss.Disease(infection=S >> I @ ss.perday(0.3)*I, outcome=I >> {H: 0.1, ss.dead: dead, R: None} @ ss.days(8), init_n=dict(I=10), name='sir')
        if by_age:
            dis.stratify(age={0: 0.25, 18: 0.6, 65: 0.15})
        return dis
    run = lambda dis, **kw: ss.Sim(dis, total_pop=10_000, dur=ss.days(200), verbose=0, solver='ode', **kw).run().results
    share = lambda r: r.died.sum()/(r.died.sum() + r.sir.H[-1] + r.sir.R[-1])
    base = run(make())
    assert np.isclose(share(base), 0.02)
    byage = run(make(dead=ss.by(age=[0.001, 0.01, 0.1]), by_age=True)) # A branch probability by group (once a crash)
    assert np.isclose(share(byage), 0.25*0.001 + 0.6*0.01 + 0.15*0.1)
    adj = run(make(by_age=True), connectors=ss.modify('sir.outcome.dead', multiply=ss.by(age=[0.1, 1, 5]))) # One destination modified by group
    assert np.isclose(share(adj), 0.02*(0.25*0.1 + 0.6 + 0.15*5))
    dead = run(make(), connectors=ss.modify('sir.outcome.dead', multiply=3))
    assert np.isclose(share(dead), 0.06)
    hosp = run(make(), connectors=ss.modify('sir.outcome.H', multiply=3)) # The others (R and ss.dead) share what is left, so deaths do not vanish
    assert np.isclose(share(hosp), 0.02*0.7/0.9)
    return


@sc.timer()
def test_infected_sources():
    sc.heading('Testing that only the states marked infected count when some are marked, even without a source of infection, and that alive exists during a compartmental run')
    S, I, R = ss.states('S I R')
    E = ss.State(infected=True) # I, the source of infection, is not marked
    d = ss.Disease(S, E, I, R, infection=S >> E @ ss.perday(0.5)*I, onset=E >> I @ ss.days(3), recovery=I >> ss.dead @ ss.days(7), init_n=dict(I=10), name='seir')
    seen = {} # The number alive read during each step
    sim = ss.Sim(d, connectors=ss.modify('seir.infection', multiply=lambda sim: seen.update({sim.ti: sim.results.alive[sim.ti]}) or 1.0), total_pop=10_000, dur=ss.days(60), verbose=0, solver='ode').run()
    tis = sorted(seen)
    assert np.allclose([seen[ti] for ti in tis], sim.results.alive[tis]) and seen[tis[-1]] < seen[0] # During the run, as after it
    assert np.isclose(sim.diseases.seir.R0(), 3.5)
    r = sim.results.seir
    assert np.allclose(r.infected, r.E) and not d.graph.states.I.infected # As said, though I is the source
    return


@sc.timer()
def test_noise():
    sc.heading('Testing noise on a rate: its multiplier has mean 1 and variance sigma²/h, sigma = 0 is no noise, it is a parameter, and the solvers that cannot draw it refuse it')
    rng = np.random.default_rng(1)
    for sigma, h in [(0.0878, 1/365.25), (0.3, 0.5)]: # A shape below 1 (He et al.'s, in years) and above it
        x = np.array([ss.kernels.noise(rng, sigma, h) for _ in range(20_000)])
        var = sigma**2/h
        assert abs(x.mean() - 1) < 4*np.sqrt(var/len(x)) and abs(x.var()/var - 1) < 0.15, (x.mean(), x.var(), var)
    assert ss.kernels.noise(rng, 0.0, 1.0) == 1.0

    def make(noise=None, solver='binomial'):
        S, I, R = ss.states('S I R')
        extra = {} if noise is None else dict(noise=noise)
        d = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3)*I, **extra), recovery=I >> R @ ss.perday(0.1), init_n=dict(I=20), name='sir')
        return ss.Sim(d, solver=solver, total_pop=20_000, unit='day', start=0, stop=120, verbose=0)
    base = make().run().results.sir
    assert np.array_equal(base.I, make(noise=0.0).run().results.sir.I) # No draws: the same run
    assert not np.array_equal(base.I, make(noise=ss.par(0.5)).run().results.sir.I)
    def fast(noise=None): # Noise on a fast rate that doesn't depend on the compartments, which sets the substeps all the same
        S, I, R = ss.states('S I R')
        extra = {} if noise is None else dict(noise=noise)
        d = ss.Disease(infection=S >> I @ ss.perday(0.3)*I, recovery=ss.Transition(I, R, rate=ss.perday(2.0), **extra), init_n=dict(I=20), name='sir')
        return ss.Sim(d, solver='binomial', total_pop=20_000, unit='week', stop=20, verbose=0)
    assert np.array_equal(fast().run().results.sir.I, fast(noise=0.0).run().results.sir.I)
    class Noisy(ss.Disease): # The class form, with the noise a parameter of the class
        S, I, R = ss.states('S I R')
        sigma = ss.par(0.0, bounds=[0, 1])
        infection = ss.Transition(S, I, rate=ss.perday(0.3)*I, noise=sigma)
        recovery = I >> R @ ss.perday(0.1)
    nz = ss.Sim(Noisy(init_n=dict(I=20), name='nz'), solver='binomial', total_pop=20_000, unit='day', stop=120, verbose=0).init()
    assert 'sigma' in nz.diseases.nz.pars and nz.diseases.nz.graph.transitions.infection.noise is nz.diseases.nz.pars.sigma # The instance's copy, keeping its name
    sim = make(noise=0.2)
    sim.init()
    assert sim.diseases.sir.pars.noise_infection.value == 0.2 # Named after its slot, so it can be fitted
    assert 'gamma white noise' in str(sim.diseases.sir.describe())
    ode = make(noise=0.2, solver='ode').run().results.sir.I
    assert np.array_equal(ode, make(solver='ode').run().results.sir.I) # Left out of the ODE
    for solver in ['tau', 'abm']: # Drawn for each leap, and as agents once a step for each group
        base = make(solver=solver).run().results.sir.I
        assert np.array_equal(base, make(noise=0.0, solver=solver).run().results.sir.I)
        assert not np.array_equal(base, make(noise=0.5, solver=solver).run().results.sir.I)
    with pytest.raises(ss.CapabilityError, match='noise='):
        make(noise=0.2, solver='gillespie').run()
    with pytest.raises(ValueError, match='noise='):
        ss.Transition(ss.State(), ss.State(), dur=ss.days(3), noise=0.1)
    with pytest.raises(ValueError, match='not an expression'): # Sigma is one value in the kernels
        ss.Transition(ss.State(), ss.State(), rate=0.1, noise=ss.par(0.04)**0.5)
    return

if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    forms = test_forms()
    names = test_names()
    time  = test_time()
    sims  = test_solvers(do_plot=do_plot)
    auto  = test_auto()
    desc  = test_describe()
    load  = test_load()
    trans = test_transfer()
    life  = test_lifetable_and_seeding()
    cross = test_cross_disease_rate()
    order = test_first_state()
    seed  = test_seeded_stay()
    disc  = test_disconnected_groups()
    junc  = test_junction()
    imm   = test_immediate()
    level = test_level_change_dwell()
    chg   = test_change_in_function()
    dwell = test_given_dwell()
    inits = test_init_prev_forms()
    nulls = test_split_null()
    dexpr = test_dwell_expressions()
    sched = test_schedule()
    split = test_split_timing()
    infst = test_infection_step()
    whole = test_wholestep()
    draws = test_compiled_draws()
    life2 = test_lifelong()
    stays = test_stays()
    casc  = test_cascade()
    cascb = test_cascade_binomial()
    state = test_state_as_result()
    ends  = test_branch_ends()
    infs  = test_infected_sources()
    noise = test_noise()
