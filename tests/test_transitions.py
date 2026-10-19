"""
Test models written as declared states and transitions, and the solvers that run them
"""

# %% Imports and settings
import sys
import types
import time
import _thread
import functools
import itertools
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

kw = dict(verbose=0)


def make_sir(name='sir', **kwargs):
    """ The two-line SIR """
    sir = ss.Disease(
        infection = ss.transmission('S -> I', beta=0.3),
        recovery  = ss.progression('I -> R', dur=10),
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
    S, I, R, W = ss.states(4)
    cholera = ss.Disease(
        infection = S >> I @ beta * W.n / (W.n + K),
        recovery  = I >> R @ ss.days(5),
        shedding  = ss.null >> W @ xi * I.n,
        decay     = W >> ss.null @ ss.days(30),
        init      = dict(I=0.001),
    )
    return cholera


@sc.timer()
def test_forms():
    sc.heading('Testing that every way of writing a model gives the same model')
    forms = sc.objdict()

    # Arrow strings
    forms.string = ss.Disease(
        infection = ss.transmission('S -> I', beta=0.05),
        recovery  = ss.progression('I -> R', dur=10),
        name = 'sir',
    )

    # Operators
    S, I, R = ss.states(3)
    infection = S >> I @ 0.05 * I
    recovery = ss.Transition(I, R, dur=10)
    forms.operator = ss.Disease(infection, recovery, name='sir')

    # Fully explicit, mixing states and their names
    S, I, R = ss.states(3)
    forms.explicit = ss.Disease(
        ss.Transition(S, I, rate=0.05 * I, name='infection'),
        ss.Transition('I', 'R', dur=10, name='recovery'),
        name = 'sir',
    )

    # A class
    class SIR(ss.Disease):
        S, I, R = ss.states('S I R')
        infection = S >> I @ 0.05 * I
        recovery = ss.Transition(I, R, dur=10)
    forms.cls = SIR()

    hashes = sc.objdict()
    for key,sir in forms.items():
        sim = ss.Sim(sir, stop=50, **kw)
        hashes[key] = sim.hash()
        assert list(sir.graph.states.keys()) == ['S', 'I', 'R']
        assert {key:p.value for key,p in sir.pars.items()} == dict(beta=0.05, dur=10)
    for key,val in hashes.items():
        assert val == hashes[0], f'Expected the "{key}" form to match the string form, but {val} ≠ {hashes[0]}'

    # A different value changes the parameter hash but not the model hash
    other = ss.Disease(infection=ss.transmission('S -> I', beta=0.08), recovery=ss.progression('I -> R', dur=10), name='sir')
    sim = ss.Sim(other, stop=50, **kw)
    assert sim.hash().model == hashes[0].model
    assert sim.hash().pars != hashes[0].pars

    # Everything that changes the results changes the hash: a modifier, a value inside a function, a literal, a numerical option, and the data
    def make(rate, mods=None, **kwargs):
        S, I, R = ss.states(3)
        return ss.Sim(ss.Disease(infection=S >> I @ rate*I, recovery=I >> R @ ss.days(10), name='sir'), connectors=mods, stop=50, **kwargs, **kw)
    def fn(val):
        return lambda sim: val
    base = make(0.05)
    assert not base.initialized and base.hash() != make(0.05, ss.modify('sir.infection', multiply=0.2)).hash() and not base.initialized
    assert base.to_spec() and not base.initialized # Neither initializes the sim, so it can still be changed and run
    assert make(fn(0.1)).hash().pars != make(fn(0.5)).hash().pars
    assert make(0.05*(1 + 1e-9)).hash() != base.hash()
    config = base.hash().config
    with ss.options.context(crn=False):
        assert make(0.05).hash().config != config
    assert base.hash().data is None and make(0.05, data=pd.DataFrame(dict(t=[10], n_I=[5]))).hash().data is not None
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
    assert sorted(sir.pars.keys()) == ['beta', 'dur_I_to_S', 'dur_recovery'] # Two durations, so each is qualified by its transition
    assert sir.infection is infection
    A, B = ss.states(2)
    first = second = A >> B @ 0.3 * B
    assert 'bound to first and second' in ss.Disease(first).graph.transitions.A_to_B.guessed.name # Which name is meant is not known, and describe() says why

    sim = ss.Sim(sir, stop=10, **kw).init()
    assert 'sir' in sim.diseases # The module is named from its variable
    assert sim.unit == 'day' # From the typed durations

    # An ss.par written in a slot is named from the slot, as a plain value is
    inline = ss.Disease(infection=ss.transmission('S -> I', beta=ss.par(0.3, bounds=[0.1, 1])), recovery=ss.progression('I -> R', dur=10))
    assert inline.pars.beta.bounds == [0.1, 1]

    # Transitions as a list, and a sum of flows as a keyword, which is an outcome
    listed = ss.Disease(transitions=[ss.transmission('S -> I', beta=0.3, name='infection'), ss.progression('I -> R', dur=10, name='recovery')], init=dict(I=0.01))
    assert list(listed.transitions.keys()) == ['infection', 'recovery']
    S, I, R = ss.states(3)
    recovery, death = I >> R @ ss.days(10), I >> ss.null @ ss.days(100)
    dis = ss.Disease(infection=S >> I @ 0.3 * I, recovery=recovery, death=death, leaving=recovery + death, init=dict(I=0.01), name='dis')
    res = ss.Sim(dis, stop=50, **kw).run().results.dis
    assert np.allclose(res.leaving, res.recovery + res.death)

    # A set of states is their sum in arithmetic, and a sum of states can be the sources of a transition
    S, Ia, Is, Ih, R = ss.states(5)
    clinical = ss.Disease(S, Ia, Is, Ih, R, infection=S >> Ia @ 0.3*(Ia + Is + Ih), worse=Ia >> Is @ ss.perday(0.2), admitted=Is >> Ih @ ss.perday(0.1),
                          recovery=(Ia + Is + Ih) >> R @ ss.perday(0.05), share=Ih/(Ia + Is + Ih), init_prev=0.01, name='clinical')
    assert [s.name for s in clinical.recovery.src] == ['Ia', 'Is', 'Ih']
    res = ss.Sim(clinical, unit='day', stop=50, **kw).run().results.clinical
    assert np.allclose(res.share[1:], (res.n_Ih/(res.n_Ia + res.n_Is + res.n_Ih))[1:])

    # One parameter shared by models built more than once, as for scenarios: registered with each, and set by name in each sim
    R0 = ss.par(2.0, bounds=[1, 5])
    def shared(name):
        S, I, R = ss.states(3)
        return ss.Disease(infection=S >> I @ R0 * 0.1 * I, recovery=I >> R @ ss.days(10), init=dict(I=0.01), name=name)
    first, second = shared('a'), shared('b')
    assert first.graph.pars.R0 is second.graph.pars.R0
    peak = lambda sim: sim.run().diseases[0].results.n_I.max()
    assert peak(ss.Sim(second, stop=60, pars=dict(R0=3.0), **kw)) > 1.5*peak(ss.Sim(second, stop=60, **kw))
    assert peak(ss.Sim([shared('c'), shared('d')], stop=60, pars=dict(R0=3.0), **kw)) == peak(ss.Sim(shared('e'), stop=60, pars=dict(R0=3.0), **kw))

    # A state with no name is an error
    with pytest.raises(ss.UnnamedState):
        states = ss.states(2)
        ss.Disease(states[0] >> states[1] @ 0.1)

    # A misspelled state is an error if states were declared
    with pytest.raises(ss.UnknownState):
        S, I = ss.states(2)
        ss.Disease(ss.Transition(S, I, rate=0.1 * I), ss.progression('I -> Q', dur=3))

    # Reserved names, undriven transitions, and bad splits
    with pytest.raises(ss.ReservedName):
        ss.Disease(step=ss.progression('I -> R', dur=3))
    with pytest.raises(ss.ReservedName): # A method of ss.Disease, not of every module
        ss.Disease(describe=ss.progression('I -> R', dur=3))
    with pytest.raises(ss.UndrivenTransition):
        ss.Disease(recovery=ss.Transition('I', 'R'))
    with pytest.raises(ss.BadBranchSplit):
        ss.Disease(a=ss.progression('I -> H', p=0.2, dur=3), b=ss.progression('I -> R', p=0.7, dur=3))
    with pytest.raises(ss.BadBranchSplit): # A branch out of the model is a transition of its own
        S, I, R = ss.states('S I R')
        ss.Transition(I, [R, ss.null], p=[0.9, 0.1], dur=10)
    with pytest.raises(ss.BadBranchSplit): # The same as a weighted destination (it was a raw TypeError)
        I >> (0.9*R + 0.1*ss.null) @ 0.1
    with pytest.raises(ss.NonlinearExposure):
        S, I = ss.states(2)
        ss.Disease(infection = S >> I @ 0.1 * I * I)

    # The named errors that no other test raises (with these, every error class is raised by some test)
    seir = lambda source=None: ss.Disease(infection=ss.transmission('S -> E', beta=ss.par(0.3, bounds=[0.1, 0.5]), source=source), latency=ss.progression('E -> I', dur=5), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='seir')
    with pytest.raises(ss.SourceNeeded): # Infectious in E or in I cannot be guessed
        ss.Sim(seir(), **kw).init()
    with pytest.raises(ss.UnknownDisease):
        ss.Sim(make_sir(), interventions=ss.vaccinate(ss.Vaccine(efficacy=0.5), at=[5], disease='flu'), **kw).init()
    cyclic = make_sir()
    late = ss.outcome(cyclic.infection, name='late')
    early = ss.outcome(late, name='early')
    late.source = early
    with pytest.raises(ss.OutcomeCycle):
        ss.Sim(make_sir(late=late, early=early), **kw).init()
    with pytest.raises(ss.DataMisaligned): # Entirely after the run
        ss.Sim(seir(source='I'), data=pd.DataFrame({'time': [500], 'n_I': [10]}), stop=100, **kw).calibrate()
    with pytest.raises(ss.NothingToFit):
        ss.Sim(make_sir(), data=pd.DataFrame({'time': [50], 'n_I': [10]}), stop=100, **kw).calibrate()

    # Typos in arrow strings, a forgotten beta, and operators that would silently drop something
    with pytest.raises(ss.UnknownState): # "l" for "I": nothing enters it
        ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('l -> R', dur=10), init_prev=0.01), **kw).init()
    with pytest.raises(ss.UnknownState): # Branches use "|"
        ss.Disease(course=ss.progression('I -> (R, D)', dur=10, p=[None, 0.1]))
    with pytest.raises(ss.UndrivenTransition):
        ss.transmission('S -> I')
    W, S, I, R = ss.states('W S I R') # The water first: people are still the states the infection moves them between
    water = ss.Disease(S >> I @ ss.perday(0.5)*W.n/(W.n + 1e5), I >> R @ ss.days(5), ss.null >> W @ 10*I.n, W >> ss.null @ ss.days(30), init=dict(I=0.001), name='water')
    assert [s.name for s in water.graph.people_states] == ['S', 'I', 'R']
    assert make_sir(beta=0.5).pars.beta.value == 0.5 # A keyword number sets the parameter of that name
    seeded = [ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), name='sir', **kwargs) for kwargs in [dict(init=dict(I=1.0)), dict(init=dict(I=0.01)), dict(init_prev=0.01)]]
    assert ss.Sim(seeded[0], total_pop=1000, **kw).init().diseases.sir.engine.y[1].sum() == 1 # 1 or more is a number of people, whether 1 or 1.0
    assert ss.Sim(seeded[1], **kw).hash() == ss.Sim(seeded[2], **kw).hash() # Two spellings of one model
    with pytest.raises(ss.UnknownParameter): # ... and is an error if there is none
        make_sir(gamma=0.1)
    S, I, R = ss.states('S I R')
    with pytest.raises(TypeError): # A second driver
        I >> R @ ss.days(10) @ 0.1
    with pytest.raises(TypeError): # Arithmetic on a duration
        I >> R @ ss.days(10) * 2
    with pytest.raises(ValueError): # Levels that do not match
        ss.by(age=[1, 2, 3]) + ss.by(age=[10, 20])
    with pytest.raises(ValueError): # Stages split a dwell time
        ss.progression('I -> R', rate=0.1, stages=3)
    branches = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), severe=ss.progression('I -> H', p=0.1, dur=5), mild=ss.progression('I -> R', dur=10), name='b')
    assert 'p = 0.9' in branches.describe(output=True) # The branch that takes the rest says so
    base = S >> I
    fast, slow = base @ 0.2, base @ 0.1
    assert fast is not slow and ss.transitions.format_expr(fast.rate) == '0.2' # A partial transition can be used twice
    two = ss.Sim(make_sir(name='a'), make_sir(name='b'), **kw).init()
    assert list(two.diseases.keys()) == ['a', 'b'] # Both models, not the second as the label

    # A model can be built with no Python frame above it, as when called from R
    S, I, R = ss.states('S I R')
    dis = without_frame(ss.Disease, S >> I @ 0.3*I, I >> R @ ss.days(10), init_prev=0.01)
    sim = without_frame(ss.Sim, dis, stop=20, verbose=0)
    assert sim.run().results.sir.n_I.max() > 0
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
    assert 'beta = 0.3 per unit' in tenths.describe(output=True) and 'every 0.1 units' in tenths.describe(output=True) # Not "per step", since a step is a tenth of a unit
    with pytest.raises(ValueError, match='no dates'): # Steps have no dates (they were days underneath)
        tenths.t.datevec
    with pytest.raises(ValueError, match='yearly cycle'):
        ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=ss.seasonal(0.3, 0.2)), recovery=ss.progression('I -> R', dur=10), init_prev=0.01), stop=100, **kw).init()
    assert ss.Sim(make_sir(), stop=100, unit='day', **kw).init().unit == 'day'
    assert ss.Sim(make_sir(), stop=10, dt=ss.weeks(1), **kw).init().unit == 'week'
    assert ss.Sim(make_sir(), start='2020-01-01', stop='2020-03-01', **kw).init().unit == 'day'
    assert ss.Sim(make_sir(), start=2020, stop=2030, **kw).init().unit == 'year'
    assert ss.Sim(make_sir(), unit='day', stop=3650, **kw).init().t.npts == 3651 # With a unit given, a large number is not a calendar year
    yearlike = ss.Sim(make_sir(), unit='day', start=2000, stop=2002, **kw).init()
    assert yearlike.t.npts == 3 and any('start=2000' in note for note in yearlike.notes) # Days, as unit= says, with a note that start looks like a year

    # The unit belongs to the model: the step does not change what its bare numbers mean, and typed values that disagree need unit=
    S, I, R = ss.states('S I R')
    typed = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init_prev=0.01, name='sir')
    daily, weekly = [ss.Sim(typed, stop=84, dt=dt, **kw).run() for dt in [ss.days(1), ss.weeks(1)]]
    assert weekly.unit == 'day' and daily.results.sir.n_I.max() > 1000 and weekly.results.sir.n_I.max() > 1000 # beta stays per day
    S, I, R = ss.states('S I R')
    with pytest.raises(ss.IncompatibleTypes):
        ss.Sim(ss.Disease(S >> I @ 0.3*I, I >> R @ ss.weeks(1), R >> S @ ss.days(100), name='mixed'), **kw).init()
    with pytest.raises(ss.IncompatibleTypes, match='rate_deaths'): # As is a bare number in demographics=
        ss.Sim(ss.Disease(S >> I @ ss.perday(0.3)*I, I >> R @ ss.weeks(1), name='mixed'), demographics=ss.death(rate=0.01), **kw).init()

    # A rate and its duration say the same thing, whatever the unit
    a = ss.Disease(recovery=ss.progression('I -> R', dur=ss.weeks(1)), init=dict(I=1.0))
    b = ss.Disease(recovery=ss.progression('I -> R', rate=ss.perday(1/7)), init=dict(I=1.0))
    res = [ss.Sim(mod, stop=30, unit='day', **kw).run().diseases[0].results.n_I for mod in [a, b]]
    assert np.allclose(res[0], res[1])
    assert np.isclose(res[0][7], 10_000*np.exp(-1), rtol=1e-4) # After seven steps, a week

    # A duration as stop= with a start date is the length of the run; a rate of 0 never ends a dwell; and ages can be any distribution
    assert ss.Sim(make_sir(), start='2020-01-01', stop=ss.days(735), **kw).init().t.stop == ss.date('2020-01-01') + ss.days(735)
    assert ss.Sim(ss.Disease(recovery=ss.progression('I -> R', rate=0), init=dict(I=1.0)), stop=10, n_agents=100, **kw).run().diseases[0].results.n_I[-1] == 100
    assert ss.Sim(make_sir(), people=ss.People(1000, age_data=ss.uniform(20, 30)), stop=2, **kw).init().people.age.min() >= 20

    # Times of entering a state are step numbers, kept as 16-bit integers when the run fits; read, they are steps as floats, NaN if never
    sir = ss.Sim(make_sir(), stop=100, unit='day', n_agents=1000, **kw).run().diseases[0]
    assert sir.ti_I.raw.dtype == np.int16 and sir.exit_I.raw.dtype == np.int8 # 2 bytes and 1, not 4 and 4
    infected = sir.ti_I.notnan.uids
    assert np.isnan(sir.ti_I[sir.S.uids]).all() and (sir.ti_I[infected] <= sir.ti).all() # Never infected is NaN, not a large negative number
    assert np.allclose(sir.ti_I.years[infected], sir.t.yearvec[sir.ti_I.raw[infected]])
    S, I, R = ss.states('S I R')
    seeded = ss.Sim(ss.Disease(S >> I @ 0*I, I >> R @ ss.days(5), name='seeded'), stop=30, unit='day', n_agents=100, **kw).init()
    seeded.diseases.seeded.S[ss.uids(np.arange(10))] = False
    seeded.diseases.seeded.I[ss.uids(np.arange(10))] = True
    assert seeded.run().results.seeded.n_I[-1] == 0 # Put in a state with no time of entering, they enter it now and leave it (they stayed for ever before)

    # With no unit, a probability is per step
    c = ss.Disease(recovery=ss.progression('I -> R', rate=ss.prob(0.05)), init=dict(I=1.0))
    assert np.isclose(ss.Sim(c, stop=10, **kw).run().diseases[0].results.n_I[10], 10_000*0.95**10) # After ten steps
    return sim


@sc.timer()
def test_solvers(do_plot=False):
    sc.heading('Testing that simple models agree between solvers')
    S, I, R, D = ss.states(4)
    models = sc.objdict(
        sir = make_sir(),
        seir = ss.Disease(
            infection = ss.transmission('S -> E', beta=0.3, source='I'),
            latency   = ss.progression('E -> I', dur=3),
            recovery  = ss.progression('I -> R', dur=10, stages=3),
            init = dict(I=0.01),
        ),
        branch = ss.Disease(
            infection = ss.transmission('S -> I', beta=0.3, source='I'),
            severe    = ss.progression('I -> H', p=0.1, dur=5),
            mild      = ss.progression('I -> R', p=0.9, dur=10),
            discharge = ss.progression('H -> R | D', dur=7, p=[0.8, 0.2]),
            dying     = ss.death('I ->', rate=0.005),
            init_prev = 0.01,
        ),
        cholera = make_cholera(),
        delay = ss.Disease( # A fixed latent period, and a minimum stay before recovery: conveyors in a compartmental run
            infection = ss.transmission('S -> E', beta=0.3, source='I'),
            latency   = ss.progression('E -> I', dur=ss.constant(3)),
            recovery  = ss.progression('I -> R', rate=0.125, after=2),
            init = dict(I=0.01),
        ),
    )
    sims = sc.objdict()
    for name,mod in models.items():
        ref = None
        for solver in ['ode', 'binomial', 'gillespie', 'abm']:
            pars = dict(n_agents=20_000) if solver == 'abm' else {}
            sim = ss.Sim(mod, total_pop=100_000, dt=0.25, stop=100, solver=solver, **pars, **kw).run()
            res = sim.diseases[0].results
            total = sum(res[f'n_{s.name}'] for s in mod.graph.people_states)
            if name != 'branch': # Which has deaths
                assert np.allclose(total, 100_000, rtol=1e-6), 'People should be conserved'
            peak = res.n_I.max()
            if ref is None:
                ref = peak
            print(f'{name:8s} {solver:9s} peak infectious: {peak:9.0f}')
            rtol = 0.1 if solver == 'abm' else 0.07 # One run of 20,000 agents: the peak varies by about 5% between seeds
            assert np.isclose(peak, ref, rtol=rtol), f'Expected {solver} to match the ODE within {rtol:.0%} for {name}, but {peak:n} vs {ref:n}'
            sims[f'{name}_{solver}'] = sim

    # The exact stochastic solver, at a smaller size, and tau-leaping for a run of too many events, unless approx=False
    sim = ss.Sim(make_sir(), total_pop=2000, stop=100, solver='gillespie', **kw).run()
    assert sim.results.sir.n_R[-1] > 1000 and sim.solver == 'gillespie'
    big = ss.Sim(make_sir(), total_pop=1e8, solver='gillespie', **kw).init()
    assert big.solver == 'tau' and any('tau leaps' in note for note in big.notes)
    assert ss.Sim(make_sir(), total_pop=1e8, solver='gillespie', approx=False, **kw).init().solver == 'gillespie'
    places = ss.geo(names=[f'p{i}' for i in range(40)], pop=np.ones(40), x=np.arange(40), y=np.zeros(40))
    assert ss.Sim(make_sir().stratify(patch=places), total_pop=1e8, solver='gillespie', **kw).init().solver == 'gillespie' # By place it is always exact, which is faster than tau leaps
    assert ss.Sim(make_sir(), total_pop=1e6, solver='gillespie', **kw).init().solver == 'gillespie' # 1.7 million events take about 0.2 s
    commute = make_sir().stratify(patch=places, mobility=ss.Mobility(ss.gravity(), away=0.1))
    assert ss.Sim(commute, total_pop=1e6, solver='gillespie', **kw).init().solver == 'tau' # The same events cost 50 times more, between 40 places that people commute between
    for old in ['tau', 'sde', 'stochastic', 'ctmc']: # Solvers that have gone, and the scheme of the ODE only
        with pytest.raises(ValueError, match='gillespie'):
            ss.Sim(make_sir(), solver=old, **kw).init()
    with pytest.raises(ValueError):
        ss.Sim(make_sir(), integrator='LSODA', **kw).init()

    # Compare with the known final size of an SIR epidemic
    sim = sims.sir_ode
    final = sim.results.sir.n_R[-1]/100_000
    R0 = 3.0
    expected = sc.findroot(lambda z: z - (1 - 0.99*np.exp(-R0*z)), 0.9) if hasattr(sc, 'findroot') else 0.9405
    assert np.isclose(final, expected, atol=0.005), f'Expected the final size to be {expected:n}, not {final:n}'

    if do_plot:
        sims.sir_abm.plot()
    return sims


@sc.timer()
def test_auto():
    sc.heading('Testing how the solver is chosen')
    sim = ss.Sim(make_sir(), stop=10, **kw).init()
    assert sim.solver == 'ode'
    agents = ss.Sim(make_sir(), stop=10, n_agents=1000, **kw)
    assert agents.init().solver == 'abm'
    assert agents.new(n_agents=None).init().solver == 'ode' # None goes back to the default
    msim = ss.Sim(make_sir(), stop=10, solver='binomial', **kw).run(n_runs=2)
    assert 'sir' in msim.results.keys() and msim.results['sir'] is msim.results.sir
    assert ss.Sim(make_sir(), networks='random', stop=10, **kw).init().solver == 'abm'
    assert ss.Sim(make_sir(), demographics=ss.Births(), start=2000, stop=2010, **kw).init().solver == 'abm'

    # A dwell time that is not exponential is run as Erlang stages, unless approx=False
    sir = ss.Disease(
        infection = ss.transmission('S -> I', beta=0.3),
        recovery  = ss.progression('I -> R', dur=ss.lognorm_ex(mean=10, std=5)),
        init_prev = 0.01,
    )
    sim = ss.Sim(sir, stop=100, **kw).run()
    assert sim.solver == 'ode'
    assert len(sim.diseases[0].engine.state_comps['I']) == 4 # (10/5)² stages
    assert any('Erlang' in note for note in sim.notes)
    assert ss.Sim(sir, stop=100, approx=False, **kw).init().solver == 'abm'
    with pytest.raises(ss.CapabilityError):
        ss.Sim(sir, stop=100, approx=False, solver='ode', **kw).init()

    # A distribution on beta varies between agents, so agents are chosen; a compartmental run uses its mean
    spread = ss.Disease(infection=ss.transmission('S -> I', beta=ss.lognorm_ex(0.3, 0.3)), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')
    auto = ss.Sim(spread, stop=100, **kw).run()
    assert auto.solver == 'abm' and 'distribution' in auto.solver_reason
    fixed = ss.Sim(make_sir(), stop=100, solver='abm', **kw).run()
    assert auto.results.sir.infection.sum() < 0.9*fixed.results.sir.infection.sum() # Varied susceptibility gives fewer infections
    ode = ss.Sim(spread, stop=100, solver='ode', **kw).run()
    assert np.allclose(ode.results.sir.n_I, ss.Sim(make_sir(), stop=100, solver='ode', **kw).run().results.sir.n_I)
    assert any('distribution across people' in note for note in ode.notes)
    with pytest.raises(ss.CapabilityError):
        ss.Sim(spread, stop=100, approx=False, solver='ode', **kw).init()

    # A module written as code cannot run as compartments
    with pytest.raises(ss.CapabilityError):
        ss.Sim(make_sir(), demographics=ss.Births(), solver='ode', start=2000, stop=2010, **kw).init()
    with pytest.raises(ss.CapabilityError): # Nor can a model with nothing declared
        ss.Sim(diseases=ss.library.AgentSIR(), networks='random', solver='ode', **kw).init()
    return sim


@sc.timer()
def test_counts():
    sc.heading('Testing states that are not people')
    cholera = make_cholera()
    graph = cholera.graph
    assert [s.name for s in graph.count_states] == ['W'] # No transition connects W to S
    assert 'people' in graph.states.W.guessed
    sim = ss.Sim(cholera, total_pop=100_000, unit='day', stop=200, **kw).run()
    assert sim.solver == 'ode' # A count never makes a model agent-based
    res = sim.results.cholera
    assert np.allclose(res.n_S + res.n_I + res.n_R, 100_000) # W is not part of the population
    assert res.n_W.max() > 1e6

    # With agents, the people are agents and the water stays a count
    sim2 = ss.Sim(cholera, total_pop=100_000, n_agents=5_000, unit='day', stop=200, **kw).run()
    res2 = sim2.results.cholera
    assert sim2.solver == 'abm'
    assert np.isclose(res2.n_W.max(), res.n_W.max(), rtol=0.2)

    # Mosquitoes given a starting number stay counts (they were the people), and ss.states() takes people=
    def vector(**kwargs):
        Sh, Ih = ss.states(2)
        Sv, Iv = ss.states(2, **kwargs)
        return ss.Disease(infection=Sh >> Ih @ ss.perday(0.1)*Iv.n/(Sh + Ih).n, recovery=Ih >> Sh @ ss.days(50), emergence=ss.null >> Sv @ ss.perday(100),
                          biting=Sv >> Iv @ ss.perday(0.1)*Ih.n/(Sh + Ih).n, mortality=Sv + Iv >> ss.null @ ss.days(10), init=dict(Sv=3000), name='vector')
    for kwargs in [{}, dict(people=False)]:
        assert [s.name for s in vector(**kwargs).graph.people_states] == ['Sh', 'Ih']

    # As agents, counts take substeps (at a monthly step they died out) and the modifiers of their transitions (which were ignored)
    for dt,mult in [(30, 1.0), (1, 2.0)]:
        sim = ss.Sim(vector(), interventions=ss.modify('vector.mortality', multiply=mult), n_agents=1000, unit='day', stop=360, dt=dt, **kw).run()
        assert np.isclose(sim.results.vector.n_Sv[-1] + sim.results.vector.n_Iv[-1], 1000/mult, rtol=0.01) # Emergence over mortality

    # A bare count in a rate is an error
    with pytest.raises(ss.ExposureToCount):
        S, I, W = ss.states(3)
        ss.Disease(infection = S >> I @ 0.5 * W, shedding = ss.null >> W @ 10 * I.n)

    # A count that is not people is a real number under every solver: a reservoir shed into by a few infected keeps them infected, as in SimInf's SISe
    def pressure():
        S, I = ss.states(2)
        phi = ss.State(people=False, name='phi')
        return ss.Disease(S, I, phi, infection=S >> I @ ss.perday(0.1)*phi.n, recovery=I >> S @ ss.perday(0.05), shed=ss.null >> phi @ ss.perday(1)*I.n/(S+I).n,
                          decay=phi >> ss.null @ ss.perday(0.1), init=dict(I=10), name='pressure')
    ode = ss.Sim(pressure(), total_pop=100, unit='day', stop=200, **kw).run().results.pressure.n_I[-1]
    for solver,agents in [('binomial', {}), ('gillespie', {}), ('abm', dict(n_agents=100))]:
        final = np.mean([ss.Sim(pressure(), total_pop=100, unit='day', stop=200, solver=solver, rand_seed=seed, **agents, **kw).run().results.pressure.n_I[-1] for seed in range(5)])
        assert np.isclose(final, ode, rtol=0.1), f'Expected {solver} to keep the infection going as the ODE does'

    # NumPy functions of counts in a rate: the same in the kernels and as agents
    def capped():
        S, I, R = ss.states(3)
        return ss.Disease(infection = S >> I @ np.clip(0.5 * I.frac, 0, 0.02) * np.exp(0 * I.n), recovery = I >> R @ ss.days(10), init=dict(I=0.01), name='capped')
    sim = ss.Sim(capped(), unit='day', stop=150, **kw).run()
    kernel = sim.results.capped.n_I.values
    agents = ss.Sim(capped(), unit='day', stop=150, n_agents=10_000, **kw).run().results.capped.n_I.values
    assert np.allclose(agents[60], kernel[60], rtol=0.1)
    assert 'minimum(maximum(0.5 * I.frac, 0), 0.02)' in str(sim.to_spec()['modules'][0]['transitions'][0]['rate'])

    # A transition for some people only
    ppl = ss.People(1)
    older = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', rate=ss.perday(0.1)),
                       vaccination=ss.progression('S -> R', rate=ss.perday(0.05), where=ppl.age >= 50), init_prev=0.01, name='older')
    older.stratify(age=[0, 50])
    for solver,agents in [('ode', {}), ('abm', dict(n_agents=5000))]:
        vaccinated = ss.Sim(older, unit='day', stop=50, solver=solver, **agents, **kw).run().results.older.vaccination.by('age').values.sum(axis=0)
        assert vaccinated[0] == 0 and vaccinated[1] > 500

    # A function of the sim that uses the model's own states (I.n) is the same as the expression written out
    def closed(fn):
        S, I, R = ss.states(3)
        rate = (lambda sim: np.clip(I.n/1000, 0, 1)*0.5) if fn else np.clip(I.n/1000, 0, 1)*0.5
        return ss.Disease(S, I, R, infection=S >> I @ 0.3*I, recovery=I >> R @ rate, init_prev=0.01, name='closed')
    assert np.allclose(*[ss.Sim(closed(fn), unit='day', stop=60, **kw).run().results.closed.n_I for fn in [True, False]])

    # where= can be a state of another module, and an arrow can have several sources
    J, A, X, Y = ss.states('J A X Y')
    growth = ss.Disease(maturity=J >> A @ ss.perday(0.1), name='growth')
    adults = ss.Disease(move=ss.Transition(X, Y, rate=ss.perday(1.0), where=growth.A), name='adults')
    assert ss.Sim([growth, adults], n_agents=100, unit='day', stop=5, **kw).run().results.adults.move.sum() > 0
    assert ss.death('M | S | R ->', rate=0.1).src == ['M', 'S', 'R'] # Resolved to states when the model is built

    # A count set each step to a value (ss.update), as odin's update() of a variable that is not a flow: the largest prevalence so far, which slows the spread, and the number infectious a step before
    def behaviour():
        S, I, R = ss.states(3)
        peak, last = ss.State(people=False), ss.State(people=False)
        return ss.Disease(infection=S >> I @ ss.perday(0.3)*I*np.exp(-5*peak.n), recovery=I >> R @ ss.days(10), track=ss.update(peak, np.maximum(peak.n, I.frac)),
                          lag=ss.update(last, I.n), init=dict(I=10), name='behaviour')
    peaks = []
    for solver in ['ode', 'binomial', 'gillespie', 'abm']:
        size = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
        res = ss.Sim(behaviour(), unit='day', stop=200, solver=solver, **size, **kw).run().results.behaviour
        frac = res.n_I.values/10_000
        assert np.allclose(res.n_peak[1:], np.maximum.accumulate(frac)[:-1]) and np.allclose(res.n_last[1:], res.n_I[:-1]), f'{solver}: the counts should be set from the step before'
        peaks.append(frac.max())
    assert np.allclose(peaks, peaks[0], rtol=0.15) and peaks[0] < 0.15 # Without the slowing, 30% are infectious at the peak

    # A comparison is 1 or 0, so a rate can stop at a capacity
    S, I, Q, R = ss.states(4)
    quarantine = ss.Disease(S, I, Q, R, infection=S >> I @ 0.3*I, isolate=I >> Q @ ss.perday(0.5)*(Q.n < 30), recovery=I >> R @ ss.days(10), release=Q >> R @ ss.days(5), init_prev=0.01, name='quarantine')
    assert 29 < ss.Sim(quarantine, unit='day', dt=ss.days(0.01), stop=60, **kw).run().results.quarantine.n_Q.max() < 30 + 0.5*10_000*0.01 # Up to one step's arrivals over it
    return sim


@sc.timer()
def test_births():
    sc.heading('Testing births and deaths, as compartments and as agents')
    def make():
        return ss.Disease(
            infection = ss.transmission('S -> I', beta=0.3),
            recovery  = ss.progression('I -> R', dur=10),
            births    = ss.birth('-> S', rate=ss.peryear(0.5)),
            deaths    = ss.death(rate=ss.peryear(0.2)),
            init      = dict(I=0.01),
            name      = 'sir',
        )
    sims = sc.objdict()
    for solver in ['ode', 'abm']:
        pars = dict(n_agents=5_000) if solver == 'abm' else {} # The agents grow past the space first set aside for them
        sims[solver] = ss.Sim(make(), unit='day', stop=365, total_pop=10_000, solver=solver, **pars, **kw).run()
    n = sc.objdict({solver:sum(sim.results.sir[f'n_{s}'][-1] for s in 'SIR') for solver,sim in sims.items()})
    assert np.isclose(n.ode, 10_000*np.exp(0.3), rtol=1e-3)
    assert np.isclose(n.abm, n.ode, rtol=0.03), f'Expected as many people as agents ({n.abm:n}) as as compartments ({n.ode:n})'
    assert np.isclose(sims.abm.results.sir.births.cum[-1], sims.ode.results.sir.births.cum[-1], rtol=0.05)

    # As agents, people who die in one disease leave the states of another on the same step
    S1, X1, S2, X2 = ss.states(4)
    one = ss.Disease(S1, X1, nothing=S1 >> X1 @ 0*S1.n, name='one')
    two = ss.Disease(S2, X2, dying=S2 >> ss.null @ ss.perday(0.1), name='two')
    res = ss.Sim([one, two], n_agents=1000, unit='day', stop=4, **kw).run().results
    assert np.allclose(res.one.n_S1, res.two.n_S2) and np.allclose(res.one.n_S1, res.n_alive)

    # Births and deaths of the population, in demographics=: the same people are born into and die from every disease, under every solver
    flu = lambda: ss.Disease(infection=ss.transmission('S -> I', beta=ss.peryear(2)), recovery=ss.progression('I -> R', rate=ss.peryear(1)), init_prev=0.05, name='flu')
    sis = lambda: ss.SIS(beta=ss.peryear(0.5), dur_inf=ss.expon(ss.years(2)), name='sis')
    demog = [ss.birth(rate=ss.peryear(0.05)), ss.death(rate=ss.peryear(0.02))]
    for solver in ['ode', 'binomial', 'gillespie', 'abm']:
        sim = ss.Sim(flu(), sis(), demographics=demog, solver=solver, n_agents=2000, stop=20, **kw).run()
        n_flu = sum(sim.results.flu[f'n_{s}'][-1] for s in 'SIR')
        n_sis = sum(sim.results.sis[f'n_{s}'][-1] for s in 'SI')
        assert np.isclose(n_flu, n_sis), f'{solver}: {n_flu:n} people in flu but {n_sis:n} in sis' # Before, births only in one disease, or twice
        assert np.isclose(n_flu, 2000*np.exp(0.03*20), rtol=0.06) # 3,644
    assert ss.Sim(flu(), demographics=demog, **kw).init().solver == 'ode' # Not agents, as birth_rate= gave
    assert ss.Sim(demographics=True, **kw).run().results.births.cumulative[-1] > 0 # With no declared disease, ss.Births and ss.Deaths
    with pytest.raises(ss.DuplicateDeclaration): # Births in two diseases would count people twice
        ss.Sim(ss.SIR(name='a', births=ss.birth(rate=0.02)), ss.SIS(name='b', births=ss.birth(rate=0.02)), **kw).init()

    # An inflow given for each level arrives at that level as given, whatever the split
    S, I = ss.states(2)
    arrivals = ss.Disease(arrive=ss.null >> S @ ss.by(sex=[100, 10]), infection=S >> I @ 0*I, name='arrivals')
    arrivals.stratify(sex=['f', 'm'], split=[1000, 500])
    res = ss.Sim(arrivals, total_pop=1500, stop=3, **kw).run().results.arrivals
    assert np.allclose(res.arrive.by('sex').values[1:], [100, 10]) # None at the start
    with pytest.raises(ValueError): # A birth rate for each level would be births from each level, which cannot be declared yet
        ss.Sim(ss.Disease(births=ss.birth('-> S', rate=ss.by(sex=[0.02, 0])), infection=ss.transmission('S -> I', beta=0), name='births').stratify(sex=['f', 'm']), **kw).init()

    # Migration: immigrants arrive at any age (and age group), and emigrants leave without dying
    for age in [False, True]:
        S, I = ss.states(2)
        mig = ss.Disease(S, I, nothing=S >> I @ 0*S.n, arrive=ss.immigration('-> S', rate=ss.peryear(500)), leave=ss.emigration(rate=ss.peryear(0.05)), name='mig')
        if age:
            mig.stratify(age=[0, 15, 50])
        sim = ss.Sim(mig, n_agents=10_000, total_pop=10_000, start=2000, stop=2010, **kw).run()
        res = sim.results
        assert res.new_deaths.sum() == 0 and res.new_emigrants.sum() == res.mig.leave.sum() > 0
        assert np.isclose(res.n_alive[-1], 10_000, rtol=0.05) # As many arrive as leave
        assert sim.people.age.mean() > 25 # Not newborns
        if age:
            assert res.mig.n_S.by('age').values[-1, -1] > 2000 # Arrivals over 50 are in the oldest group

    # An Euler step whose outflows would take more than everyone takes everyone, and no more
    S, I = ss.states(2)
    sim = ss.Sim(ss.Disease(fast=S >> I @ 5, init=dict(S=100), name='fast'), total_pop=100, start=2000, stop=2002, integrator='euler', substeps=1, **kw)
    assert np.allclose(sim.run().results.fast.n_I, [0, 100, 100]) # Everyone is infected in the first step
    return sims


@sc.timer()
def test_describe():
    sc.heading('Testing describe(), diagram(), equations(), R0(), and the transition table')
    sir = ss.Disease(
        infection = ss.transmission('S -> I', beta=0.05),
        recovery  = ss.progression('I -> R', dur=10),
    )
    sim = ss.Sim(sir, stop=100, **kw)
    text = sim.describe(output=True)
    print(text)
    for string in ['10,000 people [guessed]', 'time is in timesteps', 'inferred from the arrows', 'source is the destination',
                   'rate = 0.05 × I.n / N', 'exponential(mean = 10 steps)', 'homogeneous', 'every step: sir']:
        assert string in text, f'Expected "{string}" in describe()'
    assert '= 0.0952 per step' in sim.describe(output=True, verbose=True) # The probability of recovering in a step, 1 - exp(-1/10)
    S, I, R = ss.states(3)
    density = ss.Sim(ss.Disease(S >> I @ ss.perday(3e-5)*I.n, I >> R @ ss.days(10), init=dict(I=10)), **kw).init()
    assert 'mixing' not in density.describe(output=True) # Nothing is contact-mediated, so there is no mixing to describe (it said "frequency-dependent")

    assert sir.diagram('mermaid').splitlines() == ['flowchart LR', '    S -.->|infection| I', '    I -->|recovery| R']
    sir.diagram()

    eqs = sir.equations(output=True)
    print(eqs)
    assert eqs.splitlines() == ['dS/dt = -beta * S * I / N', 'dI/dt =  beta * S * I / N - I / dur', 'dR/dt =  I / dur']
    table = repr(sir.transitions)
    print(table)
    assert 'contact-mediated, source I' in table
    print(sim.solvers(output=True))
    spec = sim.to_spec()
    assert spec['modules'][0]['transitions'][0]['rate'] == 'beta * I'

    # R0 from the next-generation matrix: beta × the mean time infectious, with a competing death; the contact matrix's largest eigenvalue; and through a reservoir
    assert np.isclose(sir.R0(), 0.5)
    seir = ss.Disease(infection=ss.transmission('S -> E', beta=0.3, source='I'), onset=ss.progression('E -> I', dur=5),
                      recovery=ss.progression('I -> R', dur=10), death=ss.death('I', rate=0.05))
    assert np.isclose(seir.R0(), 0.3/(0.1 + 0.05))
    C = np.array([[10, 3], [3, 5]])
    aged = ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.03), recovery=ss.progression('I -> R', dur=10)), networks=ss.contacts(C, bands=[0, 20]), **kw)
    assert np.isclose(aged.R0(), 0.3*np.linalg.eigvals(C).max())
    cholera = ss.Sim(make_cholera(), total_pop=10_000, unit='day', **kw)
    assert np.isclose(cholera.R0(), 0.5/1e5*10_000*10*5*30) # Infection × shedding × time infectious × time in the water
    return sim


def test_requires():
    sc.heading('Testing requires and sim.requires()')
    class Net(ss.RandomNet):
        requires = [ss.Requires('contact_matrix', shape='age x age', source='Prem et al. 2021'), ss.Requires('n_contacts')]
    beta = ss.par(0.3, source='Smith 2020')
    dur = ss.par(None, name='dur') # Not yet known, so the sim cannot be initialized
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', dur=dur), name='sir')
    sim = ss.Sim(sir, networks=Net(name='net'), n_agents=1000, stop=10, **kw)
    df = sim.requires(output=True)
    print(df)
    status = dict(zip(df.name, df.status))
    assert status == dict(contact_matrix='missing', n_contacts='given', beta='given', dur='missing')
    assert df.set_index('name').source.beta == 'Smith 2020'
    sim.requires()
    return df


@sc.timer()
def test_load():
    sc.heading('Testing ss.load() of a model spec')
    S, I, R, W = ss.states(4)
    infection = S >> I @ ss.perday(0.5) * W.n / (W.n + 1e5)
    cholera = ss.Disease(
        infection = infection,
        recovery  = ss.progression('I -> R | S', dur=ss.lognorm_ex(5, 2, unit='days'), p=[0.9, 0.1]),
        shedding  = ss.null >> W @ ss.par(10, bounds=[5, 20], name='shed') * I.n,
        decay     = W >> ss.null @ ss.days(30),
        cases     = ss.outcome(0.5*infection, delay=ss.days(3)),
        prev      = I/(S + I + R),
        init      = dict(I=5),
    )
    cholera.stratify(age=[0, 15], adjust={'recovery': [1.5, 1]})
    sim = ss.Sim(cholera, unit='day', start='2020-01-01', stop='2020-07-01', total_pop=1000, solver='gillespie', **kw)
    sim.stratify(risk=['low', 'high'], split=[0.8, 0.2])
    ns = {} # The model as code builds the same model again
    exec(sim.to_code(), ns)
    sim2 = ns['sim']
    sim2.pars.verbose = 0
    a, b = [{k:v for k,v in s.to_spec().items() if k != 'meta'} for s in [sim, sim2]]
    assert a == b # The same spec, so the same hash
    assert sim2.hash() == sim.hash()
    sim.run()
    sim2.run()
    assert np.array_equal(sim.results.cholera.cases.values, sim2.results.cholera.cases.values)

    # A sim that has run can be saved and loaded under every solver
    for solver in ['ode', 'binomial', 'gillespie', 'abm']:
        pars = dict(n_agents=500) if solver == 'abm' else {}
        sim = ss.Sim(make_sir(), stop=50, solver=solver, **pars, **kw).run()
        sim.save('temp.sim')
        assert np.array_equal(ss.load('temp.sim').results.sir.n_I, sim.results.sir.n_I), solver
    sc.rmpath('temp.sim')

    # Rates are written with the parentheses they need, so they are read back the same
    S, I, W = ss.State(name='S'), ss.State(name='I'), ss.State(name='W')
    rate = 1 - (1 - 0.5*I.frac)**3 / (2*W.n) - (I.n - W.n)
    assert ss.transitions.format_expr(rate) == '1 - (1 - 0.5 * I.frac) ** 3 / (2 * W.n) - (I.n - W.n)'

    # A model spec is not read back, and says what to do instead
    sim.to_spec('temp_spec.json')
    with pytest.raises(ValueError, match='model.py'):
        ss.load('temp_spec.json')
    sc.rmpath('temp_spec.json')
    return sim2


@sc.timer()
def test_stratify():
    sc.heading('Testing stratification by age with a contact matrix')
    matrix = [[12, 4, 1], [3, 8, 2], [1, 3, 4]] # Contacts per day of each age group (rows) with each age group
    sims = sc.objdict()
    for solver in ['ode', 'binomial', 'abm']:
        sir = make_sir().stratify(age=[0, 15, 65], adjust={'recovery': [1.5, 1.0, 0.5]})
        sir.pars.beta.value = 0.03
        pars = dict(n_agents=20_000) if solver == 'abm' else {}
        sim = ss.Sim(sir, networks=ss.contacts(matrix, bands=[0, 15, 65]), total_pop=100_000, unit='day', stop=150, solver=solver, **pars, **kw).run()
        sims[solver] = sim
    with pytest.raises(TypeError): # The matrix's own age bands are bands=; age= is a matrix over the model's age groups
        ss.contacts(matrix, age=[0, 15, 65])
    res = sims.ode.results.sir
    by_age = res.n_S.by('age')
    assert by_age.values.shape == (151, 3)
    assert np.allclose(by_age.values.sum(axis=1), res.n_S) # The strata sum to the total
    assert np.allclose(res.n_I.select(age=0), res.n_I.by('age').values[:, 0]) # By position
    assert np.allclose(res.n_I.select(age='65+'), res.n_I.by('age').values[:, 2]) # By label
    assert np.allclose(res.n_I.select(age=[0, 1]).by('age').values[:, 2], 0) # A selection can be split again
    S, I, R = ss.states(3)
    summed = ss.Disease(S, I, R, infection=S >> I @ 0.3*I, recovery=I >> R @ ss.perday(0.1), ever=I + R, init_prev=0.01, name='summed')
    summed.stratify(age=[0, 20])
    sres = ss.Sim(summed, unit='day', stop=40, **kw).run().results.summed
    assert np.allclose(sres.ever.by('age').values, (sres.n_I + sres.n_R).by('age').values) # A declared sum of states is stratified
    S, I, R = ss.states(3)
    ratio = ss.Disease(S, I, R, infection=S >> I @ 0.3*I, recovery=I >> R @ ss.perday(0.1), prev=I/(S+I+R), init_prev=0.01, name='ratio')
    ratio.stratify(age=[0, 20, 60])
    rres = ss.Sim(ratio, unit='day', stop=40, **kw).run().results.ratio
    young = ['0-19', '20-59']
    assert np.allclose(rres.prev.select(age=young), rres.n_I.select(age=young)/(rres.n_S + rres.n_I + rres.n_R).select(age=young)) # A ratio adds up its numerator and denominator over the groups selected, then divides
    assert np.allclose(ss.by(age=0.2).per_group(sims.ode), 0.2) # One value for every level
    assert (1/ss.by(sex=[0.5, 2])).values == [2, 0.5] and np.allclose((ss.by([[1, 2]], dims=['a', 'b'])*2).array, [[2, 4]]) # Arithmetic on values by group
    pop = (res.n_S + res.n_I + res.n_R).by('age').values[0]
    world = ss.named.world
    assert np.allclose(pop, 100_000*ss.named.age_split([0, 15, 65], world.ages, world.counts), rtol=0.01) and pop[2] < 0.15*pop.sum() # With no split given, the people's ages are the world's (not uniform to 100, which put 35% over 65)
    attack = [res.infection.select(age=i).sum()/pop[i] for i in range(3)]
    assert attack[0] > attack[2], 'Children have the most contacts, so should have the highest attack rate'
    for solver in ['binomial', 'abm']:
        other = sims[solver].results.sir
        assert np.isclose(other.infection.cum[-1], res.infection.cum[-1], rtol=0.05), f'Expected {solver} to match the ODE'
        assert np.allclose(other.infection.by('age').values.sum(axis=0), res.infection.by('age').values.sum(axis=0), rtol=0.1)

    # A mixing pool of everyone and a random network with the same number of contacts agree, as agents and as an ODE
    finals = sc.objdict()
    for key,pars in dict(pool_ode=dict(networks=ss.MixingPool(n_contacts=10)), pool_abm=dict(networks=ss.MixingPool(n_contacts=10), n_agents=20_000),
                         net_abm=dict(networks=ss.RandomNet(n_contacts=10), n_agents=20_000), net_ode=dict(networks=ss.RandomNet(n_contacts=10), solver='ode'),
                         typed_ode=dict(networks=ss.MixingPool(n_contacts=ss.perweek(70))), typed_abm=dict(networks=ss.MixingPool(n_contacts=ss.perweek(70)), n_agents=20_000)).items(): # Contacts per unit time, given as a rate
        sir = make_sir()
        sir.pars.beta.value = 0.03
        sim = ss.Sim(sir, total_pop=100_000, unit='day', stop=150, **pars, **kw).run()
        finals[key] = sim.results.sir.infection.cum[-1]
        if key == 'net_ode':
            assert any('approximated' in note for note in sim.notes) # Asking for an ODE of a network model gives the standard approximation, and says so
    assert np.allclose(finals[:], finals[0], rtol=0.03), f'Expected the same attack rate from each, not {finals}'
    with pytest.raises(ss.WrongDimension): # beta and n_contacts both per unit time
        ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.03)), init_prev=0.01), networks=ss.MixingPool(n_contacts=ss.perday(10)), **kw).init()

    # Stratifying the sim gives every disease the same groups
    def make(name, beta, dur):
        return ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> S', dur=dur), init_prev=0.01, name=name)
    pars = dict(unit='day', stop=100, **kw)
    both = ss.Sim([make('flu', 0.4, 5), make('rsv', 0.3, 8)], **pars).stratify(age=[0, 5, 18, 65]).run()
    same = ss.Sim([make('flu', 0.4, 5), make('rsv', 0.3, 8)], stratify=dict(age=[0, 5, 18, 65]), **pars).init()
    assert both.hash().model == same.hash().model
    for key in ['flu', 'rsv']:
        assert both.results[key].n_I.by('age').values.shape == (101, 4)
    assert both.new().run().results.flu.n_I.by('age').values.shape == (101, 4) # A new sim is stratified too

    # Errors
    again = make_sir().stratify(age=[0, 15]).stratify(age=[0, 15]) # The same line run twice does nothing more
    assert len(again.graph.strats) == 1
    assert ss.Sim(make_sir(), **kw).stratify(age=[0, 15]).stratify(age=[0, 15]).init().n_groups == 2 # Likewise for the sim
    with pytest.raises(ss.DuplicateStructuralDimension): # In another way, it is an error
        make_sir().stratify(age=[0, 15]).stratify(age=[0, 20])
    with pytest.raises(ss.UnknownTransition):
        make_sir().stratify(age=[0, 15], adjust={'recovry': [1, 2]})
    with pytest.raises(ss.BadSplit):
        make_sir().stratify(risk=['low', 'high'], split=[0.5, 0.3, 0.2])
    with pytest.raises(ss.ShapeError): # Caught when the sim is built, not when it runs
        ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=ss.by(age=[0.3, 0.2])), init_prev=0.01), stratify=dict(age=[0, 15, 65]), **kw).init()
    with pytest.raises(ss.InconsistentDimension):
        a = make_sir().stratify(age=[0, 15])
        b = ss.Disease(recovery=ss.progression('I -> R', dur=3), name='other').stratify(age=[0, 20])
        ss.Sim([a, b], **kw).init()

    # One parser for the sim and the disease: an option is never a dimension, and the sim's adjust= is checked too
    assert [strat.name for strat in ss.Sim(make_sir(), stratify=dict(age=[0, 15], states=['S', 'I']), **kw).init().strats] == ['age'] # Not a dimension called states
    with pytest.raises(ss.UnknownTransition):
        ss.Sim(make_sir(), stratify=dict(age=[0, 15], adjust={'recovry': [1, 2]}), **kw).init()
    with pytest.raises(ValueError):
        make_sir().stratify(split=[0.5, 0.5]) # No dimension

    # A dimension called immunity, with its split used even though the sim has the same groups
    imm = ss.Sim(make_sir().stratify(immunity=['none', 'high'], split=[0.7, 0.3]), stratify=dict(immunity=['none', 'high']), unit='day', stop=10, **kw).run()
    n = sum(imm.results.sir[f'n_{k}'].by('immunity').values[-1] for k in 'SIR')
    assert np.allclose(n/n.sum(), [0.7, 0.3]) # Not no dimension, nor the sim's equal split
    return sims


@sc.timer()
def test_transfer():
    sc.heading('Testing vaccination as a stratification plus a transfer')
    sims = sc.objdict()
    for solver in ['ode', 'abm']:
        sir = make_sir()
        sir.pars.beta.value = 0.2
        sir.stratify(risk=['low', 'high'], split=[0.8, 0.2], adjust={'infection': [1.0, 3.0]})
        sir.stratify(vaccinated=[False, True], split=[1, 0], adjust={'infection': [1.0, 0.1]},
                     vaccination=ss.transfer(vaccinated=True, rate=ss.perday(0.02), states=['S']))
        pars = dict(n_agents=20_000) if solver == 'abm' else {}
        sims[solver] = ss.Sim(sir, total_pop=100_000, unit='day', stop=150, solver=solver, **pars, **kw).run()
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
    plain.stratify(risk=['low', 'high'], split=[0.8, 0.2], adjust={'infection': [1.0, 3.0]})
    base = ss.Sim(plain, total_pop=100_000, unit='day', stop=150, **kw).run()
    assert base.results.sir.infection.cum[-1] > 1.2*ode.infection.cum[-1]

    # A dimension that people are moved along, with no split given, starts with everyone at its first level
    staged = make_sir().stratify(stage=['early', 'late'], progress=ss.transfer(stage='early -> late', rate=ss.perday(0.01)))
    assert np.allclose(ss.Sim(staged, unit='day', stop=10, **kw).init().strats[0].split, [1, 0])

    # A transfer along a dimension the model does not have is an error
    with pytest.raises(ss.MissingDimension):
        ss.Sim(ss.Disease(recovery=ss.progression('I -> R', dur=3), move=ss.transfer(risk='low -> high', rate=0.1)), **kw).init()
    # A transfer given to stratify() registers its parameters too, and a transition can be called "stop"
    c = ss.par(0.0)
    S, I = ss.states(2)
    later = ss.Disease(inf=S >> I @ 0.1*I, stop=I >> S @ 0.1, init=dict(I=0.01), name='later')
    later.stratify(v=[False, True], move=ss.transfer(v=True, rate=c*0.2))
    assert 'c' in later.pars and 'stop' in later.transitions
    assert ss.Sim(later, n_agents=500, stop=10, pars=dict(c=1), **kw).run().results.later.move.sum() > 0

    # A transfer's rate can hold a parameter, set by name, and can be modified
    for solver,agents in [('ode', {}), ('abm', dict(n_agents=2000))]:
        c = ss.par(0.0)
        S, I = ss.states(2)
        dis = ss.Disease(inf=S >> I @ 0.1*I, switch=ss.transfer(v=True, rate=c*0.2), init=dict(I=0.01), name='dis')
        dis.stratify(v=[False, True])
        assert ss.Sim(dis, stop=20, solver=solver, pars=dict(c=1), **agents, **kw).run().results.dis.switch.sum() > 0
        assert ss.Sim(dis, stop=20, solver=solver, pars=dict(c=1), interventions=ss.modify(dis.switch, multiply=0), **agents, **kw).run().results.dis.switch.sum() == 0
    return sims


@sc.timer()
def test_outcomes():
    sc.heading('Testing outcomes, data, and calibration')
    def make(beta):
        beta = ss.par(beta, bounds=[0.05, 1.0])
        S, I, R = ss.states(3)
        infection = S >> I @ beta * I
        recovery = I >> R @ ss.days(10)
        cases  = ss.outcome(0.5 * infection, delay=ss.days(5))
        hosp   = ss.outcome(0.05 * infection, delay=ss.days(7), duration=ss.days(10))
        deaths = ss.outcome(0.2 * hosp, delay=ss.lognorm_ex(mean=14, std=5, unit='days'))
        return ss.Disease(infection, recovery, cases, hosp, deaths, prevalence=I/(S + I + R), init_prev=0.01, name='sir')

    pars = dict(total_pop=1e6, start='2020-01-01', stop='2020-06-01', **kw)
    truth = ss.Sim(make(0.3), **pars).run()
    res = truth.results.sir
    total = res.infection.sum()
    assert truth.unit == 'day'
    assert np.isclose(res.cases.sum(), 0.5*total, rtol=0.01) # Half of infections are reported
    assert np.isclose(res.hosp.sum(), 0.05*total, rtol=0.01)
    assert np.isclose(res.deaths.sum(), 0.01*total, rtol=0.02) # Outcomes chain
    assert np.argmax(res.hosp) == np.argmax(res.infection) + 7 # The delay
    assert res.n_hosp.max() > res.hosp.max() # The number in hospital, as well as admissions
    assert np.isclose(res.prevalence.max(), res.n_I.max()/1e6)

    # Fit beta to the weekly cases, which are compared with the total over each week
    weekly = res.cases.resample('week')
    df = sc.dataframe(date=weekly.timevec, cases=weekly.values)
    sim = ss.Sim(make(0.15), data=df, **pars)
    fit = sim.calibrate()
    fit.summarize()
    assert np.isclose(fit.pars.beta, 0.3, rtol=0.01), f'Expected to recover beta=0.3, not {fit.pars.beta}'
    assert fit.mismatch < 0.01 < fit.mismatch_before
    assert fit.converged and not sim.calibrate(method='nelder-mead', maxiter=3, scan=0, verbose=0).converged
    assert np.isclose(sim.new().init().diseases.sir.pars.beta.value, 0.15) # The original is not changed
    refit = ss.Sim(make(0.15), pars=fit.pars, **pars).run() # The best-fit values can be passed to another sim
    assert np.allclose(refit.results.sir.n_I, fit.sim.results.sir.n_I)
    named = ss.Sim(make(0.3), pars=dict(dur=ss.days(5), rand_seed=2), **pars).run() # A parameter named like a sim setting is the model's; a sim setting is the sim's
    assert named.results.sir.n_I.max() < 0.5*res.n_I.max() and named.pars.rand_seed == 2 # Before, dur= was taken as the sim's duration, and changed nothing

    short = ss.Sim(make(0.3), data=df, **(pars | dict(stop='2020-03-01'))).init()
    assert any('outside the simulated period' in note for note in short.notes) # Data that the comparison leaves out is noted

    # Mistakes in the data and the parameters are caught by name
    with pytest.raises(ss.UnknownDataColumn):
        ss.Sim(make(0.3), data=df.rename(columns=dict(cases='casess')), **pars).init()
    renamed = ss.data(df.rename(columns=dict(cases='Reported')), aliases=dict(cases='Reported'))
    assert 'cases' in ss.Sim(make(0.3), data=renamed, **pars).init().data.columns
    assert 'cases' in ss.Sim(make(0.3), **pars).new(data=renamed).init().data.columns # Or to a new sim
    with pytest.raises(ss.UnknownParameter):
        ss.Sim(make(0.3), pars=dict(betta=0.2), **pars).init()
    assert ss.transitions.did_you_mean('betta', ['sir.beta', 'sir.dur']) == '; did you mean sir.beta?' # The closest only, matched without the module's name
    with pytest.raises(ss.OutOfBounds): # A value outside its bounds
        ss.Sim(make(0.3), pars=dict(beta=5), **pars).init()
    with pytest.raises(ss.UnvaluedParameter): # A parameter with no value
        ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=ss.par()), recovery=ss.progression('I -> R', dur=10), init_prev=0.01), **pars).init()
    with pytest.raises(ss.NoData):
        ss.Sim(make(0.3), **pars).calibrate()
    with pytest.raises(ValueError, match='gof'): # Rather than the default in its place
        sim.calibrate(gof='nea')
    with pytest.raises(ss.UnknownOutcomeSource):
        ss.Sim(ss.Disease(recovery=ss.progression('I -> R', dur=3), cases=ss.outcome('infection')), **kw).init()

    # The share of a flow that is reported can be a parameter
    rho = ss.par(0.6)
    S, I, R = ss.states(3)
    infection = S >> I @ 0.3 * I
    reported = ss.Disease(infection, I >> R @ ss.days(10), ss.outcome(rho * infection, name='cases'), init_prev=0.01, name='reported')
    half = ss.Sim(reported, pars=dict(rho=0.3), stop=50, **kw).run().results.reported
    assert 'rho' in reported.pars
    assert np.isclose(half.cases.sum(), 0.3*half.infection.sum(), rtol=0.01)

    # Outcomes are stratified like the flows they are counted from, in the kernels (whole and stepped) and as agents
    def by_patch():
        S, I, R = ss.states(3)
        infection = S >> I @ 0.3 * I
        sir = ss.Disease(infection, I >> R @ 0.1, ss.outcome(0.5 * infection, name='cases', delay=ss.days(3)), ss.outcome(infection, name='hosp', duration=ss.days(5)), init=dict(I=ss.by(patch=[5, 0])), name='sir')
        sir.stratify(patch=['a', 'b'], mobility=ss.Mobility(matrix=[[0, 0.01], [0.01, 0]], kind='migrate'))
        return ss.Sim(sir, total_pop=1000, unit='day', stop=80, **kw)
    kernel, stepped = by_patch().run().results.sir, by_patch().run(profile=True).results.sir # Recording the time of each step needs the loop
    agents = by_patch().new(n_agents=1000).run().results.sir
    for res in [kernel, stepped, agents]:
        assert np.allclose(res.cases.select(patch='a') + res.cases.select(patch='b'), res.cases)
        assert np.allclose(res.n_hosp.by('patch').values.sum(axis=1), res.n_hosp)
    assert np.allclose(kernel.cases.select(patch='b'), stepped.cases.select(patch='b'))
    assert np.isclose(agents.cases.select(patch='b').sum(), kernel.cases.select(patch='b').sum(), rtol=0.3)

    # A weight for each group on the flow an outcome counts, in a whole run in the kernels
    progression = ss.progression('E -> I', rate=0.5)
    weighted = ss.Disease(infection=ss.transmission('S -> E', beta=2, source='I'), onset=progression, recovery=ss.progression('I -> R', rate=0.3),
                          notified=ss.outcome(ss.by(age=[0, 1])*progression), init=dict(I=10), name='weighted').stratify(age=['young', 'old'])
    wres = ss.Sim(weighted, stop=50, **kw).run().results.weighted
    assert np.isclose(wres.notified.sum(), wres.onset.select(age='old').sum())

    # A stay in an outcome given as a distribution: a fixed stay of ten days counts the last ten days of admissions
    for solver in ['ode', 'binomial']:
        S, I, R = ss.states(3)
        infection = S >> I @ 0.3 * I
        fixed = ss.Disease(infection, I >> R @ ss.days(10), ss.outcome(infection, name='hosp', duration=ss.constant(10, unit='days')), init_prev=0.01, name='fixed')
        res = ss.Sim(fixed, unit='day', stop=100, solver=solver, **kw).run().results.fixed
        assert np.allclose(res.n_hosp, np.convolve(res.hosp, np.ones(10))[:len(res.hosp)])

    # A share of a finished transition is a share of its flow, an outcome, never a second transition at a share of the rate
    S, I, R = ss.states(3)
    infection = S >> I @ 0.3*I.frac
    shared = ss.Disease(infection=infection, recovery=I >> R @ ss.days(10), cases=0.5*infection, init=dict(I=10), name='shared')
    res = ss.Sim(shared, unit='day', stop=60, **kw).run().results.shared
    assert list(shared.graph.transitions) == ['infection', 'recovery'] and np.allclose(res.cases, 0.5*res.infection)
    return fit


@sc.timer()
def test_likelihood():
    sc.heading('Testing likelihoods and model comparison')
    def make(beta=0.4, k=10, exposed=False, prior=None, lik=True):
        beta = ss.par(ss.perday(beta), bounds=[0.05, 1.0], prior=prior)
        k = ss.par(k, bounds=[0.1, 100])
        S, E, I, R = ss.states(4)
        infection = S >> (E if exposed else I) @ beta * I
        stages = [E >> I @ ss.days(3)] if exposed else []
        cases = ss.outcome(0.6 * infection, likelihood=ss.nbinom(n=k) if lik else None)
        N = S + E + I + R if exposed else S + I + R
        sir = ss.Disease(infection, *stages, I >> R @ ss.days(7), cases, prevalence=I/N, init=dict(I=10), name='sir')
        return ss.Sim(sir, total_pop=10_000, unit='day', stop=ss.weeks(20), label='SEIR' if exposed else 'SIR', **kw)

    # Data drawn from a negative binomial around the weekly cases of an SEIR
    truth = make(exposed=True).run().results.sir
    weekly = truth.cases.resample('week')
    rng = np.random.default_rng(1)
    df = sc.dataframe(t=weekly.timevec, cases=rng.negative_binomial(10, 10/(10 + weekly.values)))

    # The declared likelihood is used, and its dispersion is fitted with beta
    fit = make(beta=0.2, k=3, exposed=True).calibrate(data=df, method='l-bfgs-b') # The maximum likelihood (the default, NUTS, is tested in test_gradients.py)
    fit.summarize()
    assert fit.measure == 'negative log-likelihood'
    assert np.isclose(fit.pars.beta.value, 0.4, rtol=0.05)
    assert 2 < fit.pars.k < 100
    assert np.isclose(ss.loglik(fit.sim), -fit.mismatch)
    pulled = make(beta=0.2, k=3, exposed=True, prior=ss.normal(0.3, 0.002)).calibrate(data=df, method='l-bfgs-b') # With a prior, the most probable value
    assert pulled.measure == 'negative log-posterior' and pulled.pars.beta.value < 0.35

    # Counts that are not whole numbers (e.g. weekly averages) have a likelihood, and data that no run can give stops the fit
    averaged = make(beta=0.2).calibrate(data=df.assign(cases=df.cases/7), method='l-bfgs-b')
    assert np.isfinite(averaged.mismatch) and averaged.pars.beta.value > 0.2
    with pytest.raises(ValueError, match='cases'):
        make().calibrate(data=df.assign(cases=-df.cases/7), method='l-bfgs-b')

    # A likelihood can be given at calibration time, for a result that is not an outcome
    prev = sc.dataframe(t=truth.timevec[::7], prevalence=truth.prevalence.values[::7])
    fit2 = make(beta=0.2, exposed=True).calibrate(data=prev, likelihood=dict(prevalence=ss.normal(scale=0.01)), method='l-bfgs-b')
    assert fit2.measure == 'negative log-likelihood'
    assert np.isclose(fit2.pars.beta.value, 0.4, rtol=0.01)

    # The model that made the data predicts the held-out weeks better
    comp = ss.compare_models([make(), make(exposed=True)], data=df)
    assert comp.table.index[0] == 'SEIR'
    assert list(comp.fits.keys()) == ['SIR', 'SEIR']
    mixed = ss.compare_models(dict(lik=make(), nolik=make(lik=False)), data=df).table # One measure for both, although only one has a likelihood
    assert np.isclose(mixed.fit_error['lik'], mixed.fit_error['nolik'])

    # Data with its dates as text can be plotted with the sim
    dated = ss.Sim(make_sir(), start='2020-01-01', stop='2020-02-01', data=pd.DataFrame(dict(date=['2020-01-08', '2020-01-15'], infection=[5, 9])), **kw).run()
    dated.plot()

    # Parameters can be fitted inside an initial count and inside a seasonal value
    def seasonal(I0, smin):
        S, E, I, R = ss.states(4)
        season = ss.seasonal(mean=(1 + smin)/2, amplitude=(1 - smin)/(1 + smin), peak='January')
        return ss.Sim(ss.Disease(infection=S >> E @ 0.4*season*I, onset=E >> I @ ss.days(3), recovery=I >> R @ ss.days(7), init=dict(E=I0/2, I=I0), name='x'),
                      start='2020-01-01', stop='2020-07-01', unit='day', **kw)
    truth = seasonal(ss.par(80.0), ss.par(0.3)).run()
    data = pd.DataFrame(dict(date=truth.t.datevec[::7], n_I=truth.results.x.n_I.values[::7])).set_index('date')
    fit3 = seasonal(ss.par(20.0, bounds=[1, 200]), ss.par(0.7, bounds=[0.1, 1])).calibrate(data=data)
    best = fit3.table.set_index('par').best
    assert np.isclose(best.I0, 80, rtol=0.02) and np.isclose(best.smin, 0.3, rtol=0.02)
    return comp


@sc.timer()
def test_modify():
    sc.heading('Testing modifiers, scenarios, and many runs')
    sir = make_sir()
    masks = ss.modify(sir.pars.beta, multiply=0.5, start=20)
    pars = dict(stop=100, **kw)
    msim = ss.parallel(baseline=ss.Sim(sir, **pars), masks=ss.Sim(sir, interventions=masks, **pars))
    n_R = msim.results.sir.n_R
    assert [sim.label for sim in msim.sims] == ['baseline', 'masks']
    assert n_R.select(scenario='masks').values[0, -1] < 0.8*n_R.select(scenario='baseline').values[0, -1]
    assert np.allclose(n_R.select(scenario='masks').values[0, :20], n_R.select(scenario='baseline').values[0, :20]) # The same until it starts
    assert 'masks' in msim.sims[1].interventions # Named from its variable
    assert np.isclose(sir.pars.beta.value, 0.3) # The base value is never overwritten
    assert msim.sims[0].hash().scenario != msim.sims[1].hash().scenario
    def early(sim): return sim.ti > 3
    def late(sim): return sim.ti > 50
    timed = lambda when: ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.5, when=when), **pars)
    assert timed(early).hash().scenario != timed(late).hash().scenario # when= is in the spec, with its source
    assert 'when=early' in timed(early).to_code()
    with pytest.raises(AttributeError, match='typo'): # A mistake in the function is reported as itself, not as the modifier having no "active"
        timed(lambda sim: sim.results.sir.typo).run()

    # A value assigned to a parameter keeps it a parameter, so it can still be modified; one assigned on the sim's parameters is used
    sir = make_sir()
    sir.pars.beta = 0.6
    assert ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0), **pars).run().results.sir.infection.sum() == 0
    sim = ss.Sim(make_sir(), **pars)
    sim.pars.beta = 0.0
    assert sim.run().results.sir.infection.sum() == 0
    sir.pars.beta = ss.par(0.4, bounds=[0.1, 1]) # A new ss.par is taken by the parameter the transitions use
    assert sir.pars.beta is sir.graph.pars.beta and sir.graph.pars.beta.bounds == [0.1, 1]

    # As agents, a modifier on a transition with a dwell time that starts during the run reaches the people already waiting
    sir = make_sir()
    halves = {}
    for solver,agents in [('ode', {}), ('abm', dict(n_agents=10_000))]:
        base, faster = [ss.Sim(sir, stop=40, solver=solver, interventions=mods, **agents, **kw).run().results.sir.n_I[25] for mods in [None, ss.modify(sir.recovery, multiply=2, start=20)]]
        halves[solver] = faster/base
    assert abs(halves['abm'] - halves['ode']) < 0.1, f'Expected the same effect as agents and as compartments, not {halves}'

    # So do a modifier on its duration and a duration that changes in time, as agents: a cohort, 498 left in the ODE on day 25 or 15 (not 825 and 4,755)
    def cohort(solver, dur, mods=None):
        I, R = ss.states('I R')
        dis = ss.Disease(recovery=I >> R @ dur, init=dict(I=1.0), name='sir')
        size = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
        return ss.Sim(dis, interventions=mods(dis) if mods else None, unit='day', dt=0.1, stop=40, solver=solver, rand_seed=1, verbose=0, **size).run().results.sir.n_I.values
    shorter = lambda dis: ss.modify(dis.pars.dur, multiply=0.5, start=20)
    varying = ss.timeseries({0: ss.days(20), 10: ss.days(2)})
    for dur, mods, i in [(ss.days(10), shorter, 250), (varying, None, 150)]:
        ode, abm = cohort('ode', dur, mods)[i], cohort('abm', dur, mods)[i]
        assert np.isclose(abm, ode, rtol=0.1), f'agents {abm:n}, ODE {ode:n}'

    # A where= on an exit with a dwell time that becomes true while people wait: they leave once it holds, a year later
    A, B = ss.states('A B')
    aging = ss.Disease(A, B, ab=ss.Transition(A, B, dur=ss.constant(ss.years(1)), where=lambda sim: sim.people.age.raw >= 30), init=dict(A=1.0), name='aging')
    sim = ss.Sim(aging, n_agents=5000, start=2000, stop=2004, use_aging=True, verbose=0).init()
    age = sim.people.age.raw.copy()
    left = ~sim.run().diseases.aging.A.raw
    near = (age >= 29) & (age < 29.9) & sim.people.alive.raw
    assert near.any() and left[near].all() # Turned 30 within a year (all stayed before)

    # A modifier that makes a rate negative is clipped to 0, and says so
    sir = make_sir()
    clipped = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, add=-0.5, start=10), stop=30, **kw).run()
    assert any('made it negative' in note for note in clipped.notes) and clipped.results.sir.infection[12:].sum() == 0
    assert 'clipped' in clipped.what_modifies(sir.pars.beta, output=True)

    # Modifiers that depend only on time are worked out before the run, so it stays in the kernels, with the same results as step by step
    def timed():
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=ss.seasonal(mean=0.3, amplitude=0.2)), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')
        mods = [ss.modify(sir.pars.beta, multiply=0.5, start=30, stop=60), ss.change(sir.recovery, at=[40], by=[2.0]), ss.modify(sir.rel_sus, multiply=0.8, start=50)]
        return ss.Sim(sir, interventions=mods, unit='day', stop=120, **kw).init() # A yearly cycle needs a unit
    fast, slow = timed(), timed()
    assert fast.kernel_only()
    assert np.allclose(fast.run().results.sir.n_I, slow.run(profile=True).results.sir.n_I)
    assert msim.sims[0].hash().model == msim.sims[1].hash().model

    # A modifier can also be given as a path, and combines with others by multiplication
    both = [ss.modify('sir.pars.beta', multiply=0.5, start=20, name='a'), ss.modify('sir.pars.beta', multiply=0.5, start=20, name='b')]
    twice = ss.Sim(sir, interventions=both, **pars).run()
    assert twice.results.sir.n_R[-1] < n_R.select(scenario='masks').values[0, -1]
    with pytest.raises(ss.UnmodifiableTarget):
        ss.Sim(sir, interventions=ss.modify('sir.pars.betta', multiply=0.5), **pars).init()

    # One sim is one run; many runs are a MultiSim
    sim = ss.Sim(sir, solver='binomial', total_pop=1000, stop=60, **kw)
    runs = ss.parallel(sim, n_runs=8)
    n_I = runs.results.sir.n_I
    assert n_I.values.shape == (8, 61) and len(runs) == 8
    assert len(np.unique(n_I.values[:, 30])) > 1 # Each run has its own seed
    assert runs.summarize().sir_n_I.std > 0 # Summarized over the eight runs (before, over one sim: a standard deviation of 0)
    assert np.array_equal(n_I.select(run=3).values, n_I.values[3]) and np.allclose((n_I + 1).values, n_I.values + 1) # A result like any other
    low, mid, high = n_I.quantiles([0.025, 0.5, 0.975])
    assert np.all(low <= mid) and np.all(mid <= high)
    assert sim.run(n_runs=3).results.sir.n_I.values.shape == (3, 61) # The same thing, from the sim
    return msim


@sc.timer()
def test_functions():
    sc.heading('Testing functions in the model, and networks')
    def severity(sim): # Works on agents and on age groups
        return 0.1 + 0.3*(sim.people.age > 50)

    def clearance(sim, uids): # Agent-only
        return 0.2/(1 + (sim.people.age[uids] > 50))

    def seed_cluster(sim): # Custom per-step behavior
        d = sim.diseases.d
        if sim.ti == 30 and sim.solver == 'abm':
            uids = d.S.uids[:50]
            d.S[uids] = False
            d.I[uids] = True

    def make(**kwargs):
        d = ss.Disease(
            infection = ss.transmission('S -> I', beta=0.3, source='I'),
            course    = ss.progression('I -> H | R', p=severity, dur=7),
            discharge = ss.progression('H -> R', rate=0.1),
            init_prev = 0.01,
            name = 'd',
            **kwargs,
        )
        return d

    # The function is tried on groups: with no age groups it cannot run as compartments, so the model runs as agents
    sim = ss.Sim(make(seeding=seed_cluster), stop=100, unit='day', **kw).run()
    assert sim.solver == 'abm' and 'severity' in sim.solver_reason
    assert sim.results.d.n_I[-1] == sim.diseases.d.I.count() # The counts kept through the run, which the function's changes to the states are counted in, are those of the agents
    by_age = ss.Sim(make(), stop=100, unit='day', stratify=dict(age=[0, 50]), **kw).run()
    assert by_age.solver == 'ode'
    res = by_age.results.d
    assert res.n_H.select(age='50+').max() > res.n_H.select(age='0-49').max() # Older people are more likely to be hospitalized
    assert ss.Sim(make(), stop=100, unit='day', stratify=dict(age=[0, 15, 65]), **kw).init().solver == 'abm' # 50 is inside a group, so it runs as agents (it ran as an ODE, giving each group the age of its middle)
    with pytest.raises(ss.CapabilityError):
        ss.Sim(make(), stop=100, unit='day', solver='ode', **kw).run()
    def typo(sim):
        return sim.results.d['n_X'][sim.ti] # A mistake, which agents would not fix
    with pytest.raises(ValueError, match='tried on groups'):
        bad = ss.Disease(infection=ss.transmission('S -> I', beta=0.3, source='I'), course=ss.progression('I -> H | R', p=typo, dur=7), discharge=ss.progression('H -> R', rate=0.1), name='d')
        ss.Sim(bad, stop=100, unit='day', **kw).run()

    # Code of the user's own is tried on groups (a function given as an intervention, a step() of a declared disease), and compartmental= says it outright
    def knock(sim):
        if sim.ti == 20:
            sim.diseases.sir.pars.beta = 0.1
    sir = lambda: ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')
    knocked = ss.Sim(sir(), interventions=knock, stop=60, unit='day', **kw).run()
    assert knocked.solver == 'ode' and 'knock' in knocked.solver_reason # Not 10,000 agents
    assert knocked.results.sir.infection.cum[-1] < 0.7*ss.Sim(sir(), stop=60, unit='day', **kw).run().results.sir.infection.cum[-1] # It still acts
    def stop_in_place(sim):
        if sim.ti == 20:
            sim.diseases.sir.pars.beta.raw.value = 0.0 # The number inside the typed rate, changed in place
    typed = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.3)), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')
    assert ss.Sim(typed, interventions=stop_in_place, stop=60, unit='day', **kw).run().results.sir.infection[22] == 0 # Seen (it was not)
    def seed(sim): # Per-step behavior that works on groups
        if sim.ti == 3:
            sim.diseases.sir.I.n += 5
    def seed_agents(sim): # Per-step behavior that needs agents
        sim.diseases.sir.S.uids
    seeded = lambda fn: ss.Disease(infection=ss.transmission('S -> I', beta=0), recovery=ss.progression('I -> R', dur=1000), init=dict(I=0), seeding=fn, name='sir')
    sim = ss.Sim(seeded(seed), stop=6, unit='day', **kw).run()
    assert sim.solver == 'ode' and sim.results.sir.n_I[-1] > 4.9 # Called each step, not skipped by a whole run (was 0)
    assert 'seeding' in ss.Sim(seeded(seed_agents), stop=6, unit='day', **kw).init().solver_reason # Tried on groups, so run as agents
    with pytest.raises(ss.CapabilityError):
        ss.Sim(seeded(seed_agents), stop=6, unit='day', solver='ode', **kw).run()
    class Mine(ss.Disease):
        def step(self):
            return super().step()
    class Agentish(Mine):
        compartmental = False
    mine = lambda cls: cls(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='mine')
    assert ss.Sim(mine(Mine), unit='day', **kw).init().solver == 'ode'
    assert ss.Sim(mine(Agentish), unit='day', **kw).init().solver == 'abm'
    births = ss.Sim(sir(), demographics=ss.Births(), unit='day', **kw).init()
    assert births.solver == 'abm' and 'ss.birth()' in births.solver_reason
    assert births.solver_reason in births.solvers(output=True) # solvers() says what auto did

    # A function that takes uids only works on agents
    d2 = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', rate=clearance), init_prev=0.01, name='d2')
    sim2 = ss.Sim(d2, stop=100, unit='day', **kw).run()
    assert sim2.solver == 'abm' and 'clearance' in sim2.solver_reason
    assert 'no (takes uids)' in sim2.solvers(output=True)

    # A sexual network, with a transmission probability for each direction
    hiv = ss.Disease(infection=ss.transmission('S -> I', beta=dict(mf=[0.08, 0.04])), init_prev=0.05, name='hiv')
    sim3 = ss.Sim(hiv, networks=ss.MFNet(), start=2000, stop=2020, **kw).run()
    assert sim3.solver == 'abm' and 'MFNet' in sim3.solver_reason
    assert sim3.unit == 'year'
    n_I = sim3.results.hiv.n_I
    assert n_I[-1] > 1.5*n_I[0]
    monthly = ss.Disease(infection=ss.transmission('S -> I', beta=dict(mf=[0.08, 0.04])), init_prev=0.05, name='hiv')
    with pytest.warns(RuntimeWarning, match='Starsim 3'): # Bare numbers in a beta dict were probabilities per step in Starsim 3, so a verbatim port is 12x too low at monthly steps
        ss.Sim(monthly, networks=ss.MFNet(), start=2000, stop=2002, dt=ss.years(1/12), **kw).init()
    with pytest.raises(ss.RouteMismatch):
        other = ss.Disease(infection=ss.transmission('S -> I', beta=dict(needles=0.1)), init_prev=0.05, name='hiv')
        ss.Sim(other, networks=ss.MFNet(), start=2000, stop=2005, **kw).run()

    # An Erlang dwell time is exact in every solver
    erl = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=ss.erlang(k=3, mean=ss.days(10))), init_prev=0.01, name='erl')
    stg = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=ss.days(10), stages=3), init_prev=0.01, name='stg')
    r1 = ss.Sim(erl, unit='day', stop=100, **kw).run()
    r2 = ss.Sim(stg, unit='day', stop=100, **kw).run()
    assert not r1.notes # No approximation
    assert np.allclose(r1.results.erl.n_I, r2.results.stg.n_I)

    # A branch with p= and where= fires once people are in where= (before, never), and a function p= leaves the rest to the timed exit with none (before, 0.45 instead of 0.9)
    due = lambda sim: sim.people.uid.raw*0 + (sim.ti >= 5) # Everyone, from step 5
    I, R = ss.states('I R')
    gated = ss.Disease(recovery=ss.Transition(I, R, p=1.0, rate=ss.perday(1e6), where=due), init=dict(I=1.0), name='gated')
    assert ss.Sim(gated, n_agents=1000, stop=10, unit='day', verbose=0).run().results.gated.n_R[-1] == 1000
    for solver in ['ode', 'abm']:
        S, E, A, Y = ss.State(), ss.State(init=0.5), ss.State(), ss.State()
        d = ss.Disease(S=S, E=E, A=A, Y=Y, symp=ss.Transition(E, Y, p=lambda sim: 0.9, dur=ss.weeks(1)), asymp=ss.Transition(E, A, dur=ss.weeks(1)), name='d')
        res = ss.Sim(d, n_agents=10_000, total_pop=10_000, start=2000, stop=2001, dt=1/52, solver=solver, verbose=0).run().results.d
        assert np.isclose(res.symp.sum()/(res.symp.sum() + res.asymp.sum()), 0.9, atol=0.01)
    return sim


@sc.timer()
def test_time_values():
    sc.heading('Testing values that vary in time')
    def make(beta):
        return ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> S', dur=10), init_prev=0.01, name='sis')
    dates = dict(start='2020-01-01', stop='2020-06-01', **kw)

    # A time series is a step function by default, so matches the same changes made with ss.change()
    series = ss.Sim(make(ss.timeseries({'2020-01-01': 0.3, '2020-02-01': 0.1, '2020-04-01': 0.3})), **dates).run()
    sis = make(0.3)
    changed = ss.Sim(sis, interventions=ss.change(sis.pars.beta, at=['2020-02-01', '2020-04-01'], to=[0.1, 0.3]), **dates).run()
    assert series.solver == 'ode' # A value that varies in time does not make a model agent-based
    assert np.allclose(series.results.sis.n_I, changed.results.sis.n_I)
    assert series.to_spec()['modules'][0]['pars'][0]['value']['mode'] == 'constant'

    # With no dates, the times are in the sim's unit; and it can be the amount of a modifier
    sis = make(0.3)
    ramp = ss.modify(sis.pars.beta, multiply=ss.timeseries({0: 1.0, 50: 0.2}, mode='linear'))
    res = ss.Sim(sis, interventions=ramp, stop=100, **kw).run().results.sis
    assert res.n_I[:40].max() > 2*res.n_I[-1] # Transmission falls below what is needed to sustain it
    with pytest.raises(ValueError):
        ss.Sim(make(ss.timeseries({50: 1.0, 0: 0.2})), stop=10, **kw).run()

    # A spline over typed values is the same as over the numbers in the sim's unit
    spline = lambda vals: ss.Sim(make(ss.timeseries(dict(zip(['2020-01-01', '2020-03-01', '2020-05-01'], vals)), mode='spline')), unit='day', **dates).run().results.sis.n_I
    assert np.allclose(spline([ss.perday(0.3), ss.perweek(0.7), ss.perday(0.2)]), spline([0.3, 0.1, 0.2]))

    # A time series inside ss.by: the same as a function that gives the value for each group, run in the kernels or not
    def by_age(beta):
        sis = make(beta)
        sis.stratify(age=[0, 20])
        return ss.Sim(sis, stop=100, **kw).run().results.sis.n_I
    series = by_age(ss.by(age=[ss.timeseries({0: 0.3, 50: 0.1}), 0.2]))
    func = by_age(lambda sim: ss.by(age=[0.3 if sim.t.now() < 50 else 0.1, 0.2]).per_group(sim))
    assert np.allclose(series, func)

    # A rate times a trend in time or a function of the sim, in either order, also as the deaths of everyone (DuplicateDeclaration and IncompatibleTypes before)
    trend = ss.timeseries({2000: 1.0, 2005: 0.5})
    deaths = [ss.Sim(make_sir(), demographics=ss.death(rate=rate), start=2000, stop=2010, total_pop=1000, **kw).run().results.sir.deaths.values.sum()
              for rate in [ss.peryear(0.02)*trend, trend*ss.peryear(0.02), ss.peryear(0.02)*(lambda sim: 1.0)]]
    assert deaths[0] == deaths[1] and np.isclose(deaths[0], 0.75*deaths[2], rtol=0.05) # Halved for the second half

    # A seasonal value peaks once a year, under every solver
    for solver in ['ode', 'binomial', 'gillespie', 'abm']: # With this many people, gillespie takes tau leaps
        beta = ss.seasonal(mean=ss.perday(0.3), amplitude=0.5, peak='January')
        sim = ss.Sim(make(beta), start=2020, stop=2023, dt=ss.days(1), solver=solver, total_pop=20_000, **kw).run()
        n_I = sim.results.sis.n_I[366:731] # The second year
        assert np.argmax(n_I) < 45 or np.argmax(n_I) > 320, f'Expected infections to peak in the winter with {solver}'
        assert n_I.max() > 1.5*n_I.min()
    assert sim.unit == 'day'
    assert 'seasonal(mean=' in sim.describe(output=True)

    # A branch (split on entry) whose rate varies in time is a wait at that rate, with its flow recorded (before, a compartmental run moved them all at once and recorded nothing)
    for solver in ['ode', 'binomial']:
        I, H, R = ss.states('I H R')
        severe = I >> 0.25*H @ ss.par(0.2778, name='rate_severe')*ss.seasonal(mean=1.0, amplitude=0.3, peak=0.5)
        sir = ss.Disease(severe=severe, mild=I >> 0.75*R @ 0.0909, init=dict(I=1.0), name='sir')
        res = ss.Sim(sir, total_pop=10_000, stop=60, dt=0.1, unit='day', solver=solver, verbose=0).run().results.sir
        assert res.n_H[0] == 0 and np.isclose(res.severe.cum[-1], res.n_H[-1], rtol=1e-3) # Was 2,500 at the start, and a flow of 0

    # The ODE takes a value that varies smoothly at the time of each stage of a step, so a step of a day is as good as a step of three hours; with a function in the model too
    for extra in [None, lambda sim: 0.0]:
        res = []
        for dt in [1, 1/8]:
            sis = make(ss.seasonal(mean=ss.perday(0.3), amplitude=0.5, peak='January'))
            if extra is not None:
                sis = ss.Disease(infection=ss.transmission('S -> I', beta=sis.pars.beta.raw), recovery=ss.progression('I -> S', dur=10), death=ss.death('I', rate=extra), init_prev=0.01, name='sis')
            n = int(1/dt)
            res.append(ss.Sim(sis, start=2020, stop=2021, dt=ss.days(dt), total_pop=1e4, **kw).run().results.sis.n_I.values[::n]) # At the end of each day
        assert np.abs(res[0][:len(res[1])] - res[1]).max() < 1, 'Expected the ODE to take a seasonal value at each stage of a step'
    draws = ss.truncnorm(loc=2.5, scale=0.5, low=1.0, high=2.6, strict=False).rvs(1000)
    assert draws.min() >= 1 and draws.max() <= 2.6
    yearly = ss.timeseries({'2020-01-01': 1.0, '2020-07-01': 2.0}, period=ss.years(1))
    dated = ss.Sim(stop=ss.days(800), unit='day', start='2020-01-01', **kw).init()
    assert [yearly.at(year, dated) for year in [2020.1, 2020.6, 2021.1, 2021.6]] == [1, 2, 1, 2]
    curve = ss.timeseries({0: 1.0, 1: 0.0}, mode='cosine')
    assert np.isclose(curve.at(0.5, ss.Sim(stop=2).init()), 0.5) and curve.at(0.1, ss.Sim(stop=2).init()) > 0.9

    # An ODE notes where chance would matter: a start with a few infected, and a dip below one infected person
    few = ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init=dict(I=3), name='few'), unit='day', stop=50, **kw).run()
    assert any('starts with 3 infected' in note for note in few.notes)
    rounded = ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init=dict(I=10), name='few'), total_pop=1e6, n_agents=10_000, unit='day', stop=5, **kw).run()
    assert rounded.results.few.n_I[0] == 0 and any('rounds to 0' in note for note in rounded.notes) # As agents, 10 people are a tenth of an agent, which is said
    dip = ss.Disease(infection=ss.transmission('S -> I', beta=ss.timeseries({0: 0.05, 60: 0.4})), recovery=ss.progression('I -> S', dur=10), init=dict(I=20), name='dip')
    assert any('would usually die out' in note for note in ss.Sim(dip, unit='day', stop=200, **kw).run().notes)

    # A fast transition is noted when the solver takes whole steps, and check_dt() says how much the step matters
    fast = ss.Disease(infection=ss.transmission('S -> I', beta=0.6), recovery=ss.progression('I -> R', dur=4), init_prev=0.01, name='fast')
    stepped = ss.Sim(fast, solver='binomial', unit='day', stop=60, total_pop=1e6, **kw)
    df = stepped.check_dt(output=True)
    assert any('rate × dt = 0.25' in note for note in stepped.new().init().notes)
    assert 0.05 < df.change[df.result == 'n_I'].iloc[0] < 0.2
    slow = ss.Disease(infection=ss.transmission('S -> I', beta=0.6), recovery=ss.progression('I -> R', dur=4), init_prev=0.01, name='slow', dt=ss.days(2))
    own = ss.Sim(slow, solver='binomial', unit='day', dt=1, stop=60, total_pop=1e6, **kw).check_dt(output=True)
    assert own.change[own.result == 'n_I'].iloc[0] > 0.05 # A module with its own timestep takes a smaller one too
    smooth = ss.Sim(fast, solver='ode', unit='day', stop=60, total_pop=1e6, **kw)
    assert smooth.check_dt(output=True).change.max() < 1e-3 # An ODE does not depend on the timestep
    assert not smooth.new().init().notes
    quiet = ss.Sim(fast, solver='ode', stop=ss.days(60), verbose=0).init()
    assert any("unit='day'" in note for note in quiet.notes) # Kept, with nothing printed
    with sc.capture() as txt:
        ss.Sim(fast, networks=ss.RandomNet(), solver='ode', n_agents=1000, unit='day', stop=5, verbose=1).init()
    assert txt.index('solver: ode') < txt.index('RandomNet approximated') # The solver line comes before the notes it leads to
    assert 'n_agents=1000 is not used' in txt # A compartmental run has no agents
    return sim



def test_lifetable_and_seeding():
    sc.heading('Testing deaths from a life table, and seeding a number of people on a schedule')
    ages = np.arange(101)
    dx = np.clip(np.gradient(1 - np.exp(-(ages/80.0)**6)) + 1e-4, 0, None) # Most deaths in old age
    table = ss.lifetable(ages, dx)
    def make(strat=False):
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ 0.3*I.frac, I >> R @ ss.days(8), death=ss.death(lifetable=table), init=dict(I=0), name='sir')
        return sir.stratify(age=[0, 20, 40, 60, 70, 80, 90]) if strat else sir

    # As agents, each person's age at death is drawn once from the table, given their age
    ppl = ss.People(20_000, age_data=np.array([np.arange(100), np.exp(-0.02*np.arange(100))]).T)
    sim = ss.Sim(make(), people=ppl, unit='year', dt=1, stop=3, verbose=0).init()
    assert sim.solver == 'abm' # Without a stratification by age, a life table needs agents
    age = np.asarray(sim.people.age.raw[sim.people.auids])
    expected = np.sum(1 - table.survive(age + 1)/table.survive(age))
    res = sim.run().results.sir
    assert np.isclose(res.death[1], expected, rtol=0.1) and np.isclose(res.n_S[0] - res.n_S[1], res.death[1]) # Deaths in the first year, as the table gives them
    with pytest.raises(ss.CapabilityError):
        ss.Sim(make(), unit='year', stop=3, solver='ode', verbose=0).init()
    ode = ss.Sim(make(strat=True), unit='year', stop=3, solver='ode', verbose=0).run() # Stratified by age, it is a death rate for each age group
    assert ode.results.sir.death[1] > 0

    # Seeding: three people infected every 30 days for a year, chosen from everyone susceptible, as LASER's importation
    for solver in ['ode', 'abm']:
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ ss.perday(0.05)*I.frac, I >> R @ ss.days(8), init=dict(I=0), name='sir')
        seeds = ss.Delivery(move='S -> I', capacity=3, start=ss.days(30), interval=ss.days(30), stop=ss.days(365), name='importation')
        delivered = ss.Sim(sir, interventions=seeds, unit='day', stop=400, solver=solver, n_agents=10_000, verbose=0).run().results.importation.delivered.values
        assert list(np.flatnonzero(delivered)[:3]) == [30, 60, 90] and np.allclose(delivered[delivered > 0], 3)
    return sim


def test_stays_and_offspring():
    sc.heading('Testing a minimum stay (after=) and newborns (offspring=), as EMULSION has them')
    NB, W = ss.states(2)
    growth = ss.Disease(weaning=ss.Transition(NB, W, rate=ss.perday(0.5), after=ss.days(21)), init=dict(NB=1.0), name='growth')
    sim = ss.Sim(growth, unit='day', stop=40, n_agents=5000, verbose=0).run()
    n_W = sim.results.growth.n_W.values
    assert sim.solver == 'abm' and n_W[21] == 0 and n_W[22] > 0.25*5000 # Nobody leaves for 21 days, then at the rate
    binomial = ss.Sim(growth, unit='day', stop=40, total_pop=5000, solver='binomial', verbose=0).run(n_runs=20).results.growth.n_W.values # As compartments, by a conveyor of whole steps
    ode = ss.Sim(growth, unit='day', stop=40, total_pop=5000, verbose=0).run()
    assert ode.solver == 'ode' and ode.results.growth.n_W[21] == 0 < ode.results.growth.n_W[22]
    assert np.max(binomial[:, 21]) == 0 and np.isclose(np.mean(binomial[:, 22]), n_W[22], rtol=0.05) # The same as agents, step for step
    E, D, I = ss.states(3) # An escape during a minimum stay (before=), as EMULSION's
    x = ss.Disease(onset=ss.Transition(E, I, rate=ss.perday(0.3), after=ss.days(10)), death=ss.Transition(E, D, rate=ss.perday(0.5), before=ss.days(10)), init=dict(E=1.0), name='x')
    for solver in ['abm', 'ode', 'binomial']:
        size = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
        r = ss.Sim(x, unit='day', stop=30, solver=solver, verbose=0, **size).run().results.x
        assert r.n_I[10] == 0 and r.n_I[11] > 0 and r.n_D[11] == r.n_D[30] > 0, f'{solver}: nobody should become infectious before day 10, or die after it'
    S, I, R = ss.states('S I R') # A stay on a transmission is not built yet, and no engine ignores it
    for solver in ['abm', 'ode']:
        infection = ss.transmission('S -> I', beta=0.3)
        infection.after = ss.days(5)
        with pytest.raises(ss.CapabilityError):
            ss.Sim(ss.Disease(infection=infection, recovery=I >> R @ ss.days(10), init=dict(I=10), name='x'), n_agents=500, unit='day', stop=20, solver=solver, verbose=0).init()

    # A rate competing with a dwell of whole steps acts in each of them, including for people who arrived by a progression: 0.08 leave by it during a one-month dwell (not 0)
    S, X, A, B = ss.states('S X A B')
    d = ss.Disease(S, X, A, B, enter=ss.Transition(S, X, rate=ss.peryear(0.5)), timed=ss.Transition(X, A, dur=ss.constant(ss.months(1))), hazard=ss.Transition(X, B, rate=ss.peryear(1.0)), name='d')
    r = ss.Sim(d, n_agents=20_000, start=2000, stop=2010, dt=ss.months(1), verbose=0).run().results.d
    share = r.hazard.values.sum()/(r.hazard.values.sum() + r.timed.values.sum())
    assert np.isclose(share, 1 - np.exp(-1/12), rtol=0.1)

    G, L, NB = ss.states(3)
    herd = ss.Disease(farrowing=ss.Transition(G, L, dur=ss.days(10), offspring=(NB, 3)), weaning=ss.Transition(L, G, dur=ss.days(20)), init=dict(G=1.0), name='herd')
    res = ss.Sim(herd, unit='day', stop=60, n_agents=100, verbose=0).run().results.herd
    assert res.n_NB[-1] == 3*res.farrowing.values.sum() > 0 # Three newborns for each farrowing, who are people
    return sim


def test_found_by_docs():
    sc.heading('Testing what porting the docs found: everyday calls that failed')
    import pickle
    assert pickle.loads(pickle.dumps(ss.time.UnitFactors(dict(day=1.0, year=365.0), 'year'))).day == 1/365 # Parallel workers unpickle these
    assert 'init' in ss.Sim(ss.SIR(), stop=50, verbose=0).describe(output=True) # init_prev is an ss.par
    ss.SIR().equations() # Before the disease is in a sim
    seasonal = ss.seasonal(mean=ss.peryear(0.4), amplitude=0.6, peak='January')
    assert ss.Sim(ss.SIS(beta=seasonal), start=2000, stop=2010, n_agents=500, verbose=0).run().results.sis.n_I.max() > 0 # A value that varies in time, as a keyword
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ 2*ss.seasonal(mean=0.15, amplitude=0.2, peak=0.1)*I.frac, recovery=I >> R @ ss.days(10), init=dict(I=10), name='sir') # A number times a time value
    assert ss.Sim(sir, unit='day', stop=60, verbose=0).run().results.sir.n_I.max() > 100

    # People given to a compartmental run give its age split and size, as they would as agents
    ppl = lambda: ss.People(10_000, age_data=np.array([np.arange(80), np.exp(-0.03*np.arange(80))]).T)
    def young(solver):
        sir = make_sir().stratify(age=[0, 15, 65])
        n_S = ss.Sim(sir, people=ppl(), unit='day', stop=10, solver=solver, verbose=0).run().results.sir.n_S
        return n_S.by('age').values[0, 0]/n_S[0]
    assert np.isclose(young('binomial'), young('abm'), atol=0.02) # About 40% under 15

    # And: a parameter inside a likelihood is named for its outcome; float() of a parameter; times with no calendar are numbers; a mixing pool of some people needs agents
    S, I, R = ss.states(3)
    infection = S >> I @ 0.3*I
    cases = ss.outcome(0.2*infection, likelihood=ss.nbinom(n=ss.par(10, bounds=[1, 100])))
    assert 'cases_n' in ss.Disease(infection, I >> R @ ss.days(10), cases, name='sir').graph.pars
    assert float(ss.par(0.2)) == 0.2
    df = ss.Sim(make_sir(), stop=10, verbose=0).run().results.sir.to_df()
    assert df.timevec.iloc[1] == 1 # Not a date in year 1
    pool = ss.MixingPool(src=lambda sim: sim.people.age < 15, dst=lambda sim: sim.people.age >= 15)
    assert ss.Sim(make_sir(), networks=pool, stop=10, verbose=0).init().solver == 'abm'
    k = ss.par(10, bounds=[1, 100], name='k') # A likelihood is written into the code too, so the model it builds is the same
    S, I, R = ss.states(3)
    infection = S >> I @ 0.3*I.frac
    sim = ss.Sim(ss.Disease(infection=infection, recovery=I >> R @ ss.days(10), cases=ss.outcome(0.5*infection, likelihood=ss.nbinom(n=k)), init=dict(I=10), name='sir'), stop=30, verbose=0)
    ns = {}
    exec(sim.to_code(), ns)
    assert ns['sim'].hash() == sim.hash()
    msim = ss.parallel(ss.Sim(make_sir(), stop=30, verbose=0), pars={'sir.beta': [0.2, 0.4]})
    assert msim.results.sir.n_I.select(beta=0.4).values.max() > msim.results.sir.n_I.select(beta=0.2).values.max() # A parameter swept by its full name, selected by its own
    return sir


def test_cross_disease_rate():
    sc.heading('Testing a rate that counts another disease\'s state')
    def final(cross, solver):
        syph = ss.Disease(infection=ss.transmission('S -> I', beta=0.0), recovery=ss.progression('I -> R', rate=ss.peryear(1e-12)), init=dict(I=2000), name='syph') # 10% infected throughout
        beta = 0.15*(1 + 5*syph.I.n/20_000) if cross else 0.225 # So the same
        hiv = ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', dur=10), init=dict(I=100), name='hiv')
        kw = dict(n_agents=20_000) if solver == 'abm' else {}
        return ss.Sim([syph, hiv], total_pop=20_000, stop=10, solver=solver, verbose=0, **kw).run().results.hiv.n_I[-1]
    for solver in ['ode', 'abm']:
        assert np.isclose(final(True, solver), final(False, solver), rtol=1e-9) # syph.I is syphilis's, not HIV's own I

    # The same for a condition: where=a.C is the other disease's C, whether or not this one has a C too
    def cancelled(names, solver):
        S, I, C = ss.states('S I C')
        a = ss.Disease(infection=S >> I @ ss.peryear(2)*I, cancer=I >> C @ ss.years(2), init_prev=0.2, name='a')
        S, I, C, X = ss.states(names)
        b = ss.Disease(infection=S >> I @ ss.peryear(2)*I, cancer=I >> C @ ss.years(2), cancel=ss.Transition(I, X, rate=ss.peryear(4), where=a.C), init_prev=0.2, name='b')
        return ss.Sim([a, b], n_agents=2000, total_pop=2000, stop=2005, solver=solver, verbose=0).run().results.b.cancel.sum()
    for solver in ['ode', 'abm']:
        assert np.isclose(cancelled('S I C X', solver), cancelled('S I D X', solver), rtol=1e-9) # Was 0 (b's own C) with agents, and an error with the ODE

    # A state of the other disease read before that disease is declared: a would keep an empty state of its own, so it is refused
    Sa, Ia, Sb, Ib, Rb = ss.states('Sa Ia Sb Ib Rb')
    a = ss.Disease(infection=Sa >> Ia @ ss.peryear(1.5)*Ia*(1 - 0.9*Rb.frac), name='a')
    with pytest.raises(ss.DuplicateDeclaration):
        ss.Disease(infection=Sb >> Ib @ ss.peryear(1.5)*Ib, recovery=Ib >> Rb @ ss.years(1), name='b')
    return


def test_anchor_order():
    sc.heading('Testing that the people left over start in the first state whatever order the states were made in')
    for solver in ['ode', 'abm']:
        R, I, S = ss.states(3)
        sir = ss.Disease(infection=S >> I @ 0.3*I.frac, recovery=I >> R @ ss.days(10), init=dict(I=100), name='sir')
        sim = ss.Sim(sir, total_pop=10_000, n_agents=10_000, stop=1, unit='day', solver=solver, verbose=0).run()
        assert sim.results.sir.n_S[0] == 9900 and sim.results.sir.n_R[0] == 0

    # Nor on the order of the transitions, and the code written for it says which state it is (before, a breakthrough given first started everyone in V)
    def make(**kw):
        S, I, V = ss.states('S I V')
        trs = dict(infection=S >> I @ ss.perday(0.3)*I, vax=S >> V @ ss.perday(0.01), breakthrough=V >> I @ ss.perday(0.1)*I)
        sir = ss.Disease(**{k:trs[k] for k in reversed(trs)}, init=dict(I=100), name='sir', **kw)
        return ss.Sim(sir, total_pop=10_000, stop=10, unit='day', solver='ode', verbose=0)
    sim = make()
    rebuilt = types.SimpleNamespace()
    exec(sim.to_code(), rebuilt.__dict__)
    assert "first = 'S'" in sim.to_code() and list(rebuilt.sir.graph.transitions.keys()) == ['breakthrough', 'vax', 'infection'] # In the order given
    assert sim.run().results.sir.n_S[0] == rebuilt.sim.run().results.sir.n_S[0] == 9900 # Was 0 after the round trip
    assert make(first='V').run().results.sir.n_V[0] == 9900 # Or given
    return

if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    forms = test_forms()
    names = test_names()
    time  = test_time()
    sims  = test_solvers(do_plot=do_plot)
    auto  = test_auto()
    count = test_counts()
    birth = test_births()
    desc  = test_describe()
    reqs  = test_requires()
    load  = test_load()
    strat = test_stratify()
    trans = test_transfer()
    fit   = test_outcomes()
    comp  = test_likelihood()
    msim  = test_modify()
    funcs = test_functions()
    times = test_time_values()
    life  = test_lifetable_and_seeding()
    stays = test_stays_and_offspring()
    docs  = test_found_by_docs()
    cross = test_cross_disease_rate()
    order = test_anchor_order()
