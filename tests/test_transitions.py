"""
Test models written as declared states and transitions, the solvers that run them, and treatments that move people along transitions with a rate of 0
"""

# %% Imports and settings
import warnings
from unittest import mock
import pytest
import numpy as np
import scipy.stats as stats
import sciris as sc
import starsim as ss
import corpus
from corpus import make_sir

kw = dict(verbose=0)


def same(*sims):
    """ Whether sims give the same results: two spellings of one model """
    return ss.debugtools.check_sims_match(*[sim.run() for sim in sims])


def size(solver, n=10_000):
    """ The agents or the people of a run """
    return dict(n_agents=n) if solver == 'abm' else dict(total_pop=n)


def test_forms():
    sc.heading('Testing that every way of writing a model gives the same model')
    forms = sc.obj()
    S, I, R = ss.states(3)
    infection, recovery = S >> I @ 0.05 * I, ss.Transition(I, R, dur=10) # Named from their variables
    forms.operator = ss.Disease(infection, recovery, name='sir')
    S, I, R = ss.states(3)
    forms.explicit = ss.Disease(ss.Transition(S, I, rate=0.05 * I, name='infection'), ss.Transition(I, R, dur=10, name='recovery'), name='sir')
    class SIR(ss.Disease):
        S, I, R = ss.states('S I R')
        infection = S >> I @ 0.05 * I
        recovery = ss.Transition(I, R, dur=10)
    forms.cls = SIR()
    S, I, R = ss.states(3)
    infection, recovery = S >> I @ 0.05 * S * I / (S + I + R), ss.Transition(I, R, dur=10) # As most other frameworks write it
    forms.flow = ss.Disease(infection, recovery, name='sir')
    for sir in forms.values():
        assert list(sir.states.keys()) == ['S', 'I', 'R']
        assert {key:p.value for key,p in sir.pars.items()} == dict(beta=0.05, dur_I=10, init_prev=0.01) # init_prev by default
    assert same(*[ss.Sim(sir, stop=50, **kw) for sir in forms.values()])

    # Several sources as a sum of states
    S, I, R, D = ss.states(4)
    death = ss.Transition(I + R, D, rate=0.01, name='death')
    assert [s.name for s in ss.Disease(S >> I @ 0.05*I, I >> R @ 10, death, name='sir').graph.transitions.death.source] == ['I', 'R']

    # Density-dependent: the whole flow without the total is the count
    density = []
    for flow in [True, False]:
        S, I, R = ss.states(3)
        density.append(ss.Sim(ss.Disease(S >> I @ (3e-5*S*I if flow else 3e-5*I.n), I >> R @ 10, name='sir'), stop=50, **kw))
    assert same(*density)

    # Frequency-dependent: beta*I/N is the share infectious, as is the whole flow beta*S*I/N; well mixed, the same as contact, beta*I
    freq = []
    for rate in [lambda S,I,N: 0.3*I, lambda S,I,N: 0.3*I.frac, lambda S,I,N: 0.3*I/N, lambda S,I,N: 0.3*S*I/N, lambda S,I,N: 0.3*I.n/(S.n + I.n + R.n)]:
        S, I, R, N = ss.states('S I R', include_sum=True)
        freq.append(ss.Sim(ss.Disease(S >> I @ rate(S, I, N), I >> R @ 10, name='sir'), stop=50, **kw))
    assert same(*freq)

    # On a network, only contact can be used
    S, I, R = ss.states(3)
    with pytest.raises(ValueError):
        ss.Sim(ss.Disease(S >> I @ 0.3*I.frac, I >> R @ 10, name='sir'), networks=ss.RandomNet(), n_agents=100, stop=5, verbose=0).init()


def test_names():
    sc.heading('Testing how transitions and parameters get their names, shared parameters, and seeding')
    beta = ss.par(0.3, bounds=[0.1, 0.5])
    S, I, R = ss.states(3)
    infection = S >> I @ beta * I
    recovery = I >> R @ ss.days(10)
    sir = ss.Disease(infection, recovery, I >> S @ ss.days(100))
    assert list(sir.graph.transitions.keys()) == ['infection', 'recovery', 'I_to_S'] # From their variables, else their endpoints
    assert sorted(sir.pars.keys()) == ['beta', 'dur_I_to_S', 'dur_recovery', 'init_prev'] # Two durations out of I, so each is qualified by its transition
    S, E, I, R = ss.states('S E I R')
    seir = ss.Disease(S >> E @ 0.3 * I, E >> I @ ss.days(5), I >> R @ ss.days(10), I >> S @ 0.01)
    assert sorted(seir.pars.keys()) == ['beta', 'dur_E', 'dur_I', 'init_prev', 'rate_I_to_S'] # A dwell time is named by its state, and a rate by its transition
    sim = ss.Sim(sir, stop=10, **kw).init()
    assert 'sir' in sim.diseases and sim.unit == 'day' # The module is named from its variable, and the unit from the typed durations

    # One parameter shared by models built more than once, as for scenarios; transitions reused in a second disease are copied with their parameters
    R0 = ss.par(2.0, bounds=[1, 5])
    def shared(name):
        S, I, R = ss.states(3)
        return ss.Disease(infection=S >> I @ R0 * 0.1 * I, recovery=I >> R @ ss.days(10), init=dict(I=0.01), name=name)
    peak = lambda sim: sim.run().diseases[0].results.I.max()
    assert peak(ss.Sim([shared('c'), shared('d')], stop=60, pars=dict(R0=3.0), **kw)) == peak(ss.Sim(shared('e'), stop=60, pars=dict(R0=3.0), **kw)) > 1.5*peak(ss.Sim(shared('f'), stop=60, **kw))
    S, E, I, R = ss.states('S E I R')
    inf, prog, rec = S >> E @ 1.9*I, E >> I @ ss.days(10), I >> R @ ss.days(8)
    first, second = ss.Disease(inf, prog, rec), ss.Disease(inf, prog, rec, beta=1.0, name='second')
    assert first.pars.beta.value == 1.9 and second.pars.beta.value == 1.0

    # Mistakes: in building, and in using
    with pytest.raises(ValueError): # Branches that don't sum to 1
        I, H, R = ss.states('I H R')
        ss.Disease(a=ss.Transition(I, H, p=0.2, dur=3), b=ss.Transition(I, R, p=0.7, dur=3))
    with pytest.raises(KeyError): # No such parameter
        make_sir(gamma=0.1)
    with pytest.raises(KeyError): # No such disease
        ss.Sim(make_sir(), interventions=ss.vaccinate(ss.Vaccine(efficacy=0.5), when=[5], disease='flu'), **kw).init()

    # Seeding: init= is a share, init_n= a number of people; a given S is kept and the rest go to the only state not given
    def seeded(**kwargs):
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), name='sir', **kwargs)
    assert ss.Sim(seeded(init=dict(I=1.0)), total_pop=1000, **kw).init().diseases.sir.engine.y[1].sum() == 1000
    assert ss.Sim(seeded(init_n=dict(I=1)), total_pop=1000, **kw).init().diseases.sir.engine.y[1].sum() == 1
    assert same(ss.Sim(seeded(init=dict(I=0.01)), **kw), ss.Sim(seeded(init_prev=0.01), **kw))
    S, I, R = ss.states('S I R')
    given = ss.Sim(ss.Disease(S >> I @ ss.perday(0.5)*I, I >> R @ ss.days(7), init=dict(S=0.1, I=1e-4), name='sir'), total_pop=10_000, **kw).init().diseases.sir.engine.y
    assert np.isclose(given[0].sum(), 1000) and np.isclose(given[2].sum(), 8999)


def test_time():
    sc.heading('Testing how the sim gets its unit, and times of entering a state')
    sim = ss.Sim(make_sir(), stop=100, **kw).init()
    assert sim.unit is None and sim.t.npts == 101 # Nothing has a unit, so time is in timesteps
    assert ss.Sim(make_sir(), stop=10, dt=ss.weeks(1), **kw).init().unit == 'week'
    assert ss.Sim(make_sir(), start=2020, stop=2030, **kw).init().unit == 'year'
    assert ss.Sim(make_sir(), unit='day', start=2000, stop=2002, **kw).init().t.npts == 731 # A bare number from 1900 on as the start is a calendar year, whatever the unit
    S, I, R = ss.states('S I R')
    with pytest.raises(TypeError): # Typed values that disagree leave the unit of the bare numbers unknown
        ss.Sim(ss.Disease(S >> I @ 0.3*I, I >> R @ ss.weeks(1), R >> S @ ss.days(100), name='mixed'), **kw).init()

    # A rate and its duration say the same thing; a number times a probability scales its rate; a probability of 1 moves everyone in the step, under every solver
    res = []
    for driver in [dict(dur=ss.weeks(1)), dict(rate=ss.perday(1/7))]:
        I, R = ss.states('I R')
        res.append(ss.Sim(ss.Disease(recovery=ss.Transition(I, R, **driver), default_state='I'), stop=30, unit='day', **kw).run().diseases[0].results.I)
    assert np.allclose(res[0], res[1]) and np.isclose(res[0][7], 10_000*np.exp(-1), rtol=1e-4)
    S, I = ss.states('S I')
    res = ss.Sim(ss.Disease(S, I, onset=S >> I @ ss.probperyear(0.8)*0.2, name='d'), total_pop=10_000, solver='ode', start=2000, stop=2001, **kw).run().results.d
    assert np.isclose(res.I[-1], 10_000*(1 - 0.2**0.2), rtol=1e-3)
    for solver in ['abm', 'ode', 'binomial']:
        res = ss.Sim(ss.Disease(S, I, onset=S >> I @ ss.probperyear(1)*0.8, name='d'), n_agents=1000, total_pop=1000, solver=solver, start=2000, stop=2001, **kw).run().results.d
        assert res.S[1] == 0 and res.I[1] == res.S[0] + res.I[0], solver

    # Times of entering a state are step numbers (ss.TimeArr), NaN if never, with the seeds dated a step before the start; kept only where needed, unless keep_times=True
    sir = ss.Sim(make_sir(), stop=100, unit='day', n_agents=1000, solver=ss.solvers.abm(keep_times=True), **kw).run().diseases[0]
    infected = sir.ti_I.notnan.uids
    assert isinstance(sir.ti_I, ss.TimeArr) and np.isnan(sir.ti_I[sir.S.uids]).all() and (sir.ti_I[infected] == -1).sum() == sir.results.I[0]
    assert sir.recovery.n_times.values.sum() == sir.results.recovery.sum() and sir.ti_R.notnan.uids.size > 0
    with pytest.raises(AttributeError):
        ss.Sim(make_sir(), stop=10, n_agents=100, **kw).run().diseases[0].ti_R.notnan


def test_solvers():
    sc.heading('Testing the solvers by name, the ODE against known answers, and how the solver is chosen')
    sim = ss.Sim(make_sir(), total_pop=2000, stop=100, solver='gillespie', **kw).run()
    assert sim.results.sir.R[-1] > 1000 and isinstance(sim.solver, ss.solvers.gillespie)
    tau = ss.Sim(make_sir(), total_pop=1e8, stop=100, solver='tau', **kw).run()
    assert isinstance(tau.solver, ss.solvers.tau) and tau.results.sir.R[-1] > 5e7
    sim = ss.Sim(make_sir(), total_pop=100_000, dt=0.25, stop=100, solver='ode', **kw).run()
    assert np.isclose(sim.results.sir.R[-1]/100_000, 0.9405, atol=0.005) # The final size: the root of z = 1 - 0.99*exp(-3*z)
    from scipy.integrate import solve_ivp # The default ODE follows a fast epidemic at a daily step (K&R program 2.1)
    exact = solve_ivp(lambda t, y: [-1.4247*y[0]*y[1], 1.4247*y[0]*y[1] - 0.14286*y[1]], (0, 70), [1-1e-6, 1e-6], t_eval=np.arange(71.), method='DOP853', rtol=1e-12, atol=1e-15).y[1]
    S, I, R = ss.states(3)
    kr21 = ss.Disease(infection=S >> I @ ss.perday(1.4247) * I, recovery=I >> R @ ss.perday(0.14286), init=dict(I=1e-6), name='sir')
    n_I = ss.Sim(kr21, unit='day', stop=70, total_pop=1.0, **kw).run().results.sir.I.values
    assert np.abs(n_I - exact).max()/exact.max() < 1e-4

    # How the solver is chosen: agents for n_agents, a network, code, a distribution across people, or a clock; compartments otherwise
    assert isinstance(ss.Sim(make_sir(), stop=10, n_agents=1000, **kw).init().solver, ss.solvers.abm)
    assert isinstance(ss.Sim(make_sir(), networks='random', stop=10, **kw).init().solver, ss.solvers.abm)
    assert isinstance(ss.Sim(make_sir(), demographics=ss.Births(), start=2000, stop=2010, **kw).init().solver, ss.solvers.abm)
    sir = make_sir(dur=ss.lognormal(mean=10, std=5)) # Run as Erlang stages, unless approximations='error'
    sim = ss.Sim(sir, stop=100, **kw).run()
    assert isinstance(sim.solver, ss.solvers.ode) and len(sim.diseases[0].engine.state_comps['I']) == 4
    with ss.options.context(approximations='error'):
        assert isinstance(ss.Sim(sir, stop=100, **kw).init().solver, ss.solvers.abm)
    spread = make_sir(beta=ss.lognormal(0.3, 0.3)) # A compartmental run uses its mean
    assert isinstance(ss.Sim(spread, stop=100, **kw).init().solver, ss.solvers.abm)
    assert np.allclose(ss.Sim(spread, stop=100, solver='ode', **kw).run().results.sir.I, ss.Sim(make_sir(), stop=100, solver='ode', **kw).run().results.sir.I)
    pool = ss.MixingPool(source=ss.person.age < 15, target=ss.person.age >= 15)
    assert isinstance(ss.Sim(make_sir(), networks=pool, stop=10, **kw).init().solver, ss.solvers.abm)
    with pytest.raises(ss.CapabilityError): # What cannot run as compartments says so
        ss.Sim(make_sir(), networks=pool, solver='ode', stop=10, **kw).init()
    class FadingSIR(ss.Disease): # A modifier that is a member of a disease, and reads a clock
        S, I, R = ss.states('S I R')
        infection = S >> I @ 0.5*I
        recovery  = I >> R @ ss.days(10)
        fading    = ss.modify('rel_trans', multiply=0.5**(I.time_in/ss.days(3)), who=I)
    assert isinstance(ss.Sim(FadingSIR(init_prev=0.01), total_pop=1000, stop=ss.days(10), **kw).init().solver, ss.solvers.abm)


def test_describe():
    sc.heading('Testing describe(), diagram(), equations(), and R0()')
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.05 * I), recovery=ss.Transition(I, R, dur=10))
    assert 'Every step: sir' in ss.Sim(sir, stop=100, **kw).describe()
    assert sir.diagram('text').splitlines()[0] == 'flowchart LR' # Mermaid, a documented format
    assert sir.equations().splitlines() == ['dS/dt = -beta * S * I / N', 'dI/dt =  beta * S * I / N - I / dur_I', 'dR/dt =  I / dur_I']
    S, E, I, R, H = ss.states('S E I R H') # Branches take their shares in the equations
    p_death, p_hosp = ss.par(0.002), ss.par(0.1)
    branched = ss.Disease(S >> E @ 1.9*I, E >> I @ ss.days(10), I >> H @ dict(dur=ss.days(8), p=p_hosp), I >> R @ dict(dur=ss.days(8), p=None),
                          I >> ss.dead @ dict(dur=ss.days(8), p=p_death), H >> R @ ss.days(5), init=dict(I=0.001))
    assert '(1 - p_hosp - p_death) * E / dur_E' in branched.equations()

    # R0 from the next-generation matrix: with a competing death, at the disease-free state, and with a contact matrix and a reservoir
    assert np.isclose(sir.R0(), 0.5)
    S, E, I, R = ss.states('S E I R')
    seir = ss.Disease(infection=ss.Transition(S, E, rate=0.3 * I), onset=ss.Transition(E, I, dur=5), recovery=ss.Transition(I, R, dur=10), death=ss.death(I, rate=0.05))
    assert np.isclose(seir.R0(), 0.3/(0.1 + 0.05))
    S, I, R = ss.states('S I R')
    immune = ss.Sim(ss.Disease(S >> I @ 1.9*I, I >> R @ ss.days(8), init=dict(R=0.9, I=1e-3), name='measles'), unit='day', solver='ode', verbose=0).init()
    assert np.isclose(immune.diseases.measles.R0(), 15.2) # With no immunity, whoever starts immune
    C = np.array([[10, 3], [3, 5]])
    assert np.isclose(ss.Sim(make_sir(beta=0.03), networks=ss.contacts(C, ages=[0, 20]), **kw).init().diseases[0].R0(), 0.3*np.linalg.eigvals(C).max())
    assert np.isclose(corpus.cholera().init().diseases[0].R0(), 0.5/1e5*9_990*10*5*30) # Infection × shedding × time infectious × time in the water


def test_load():
    sc.heading('Testing ss.save() and ss.load()')
    S, I, R = ss.states(3)
    W = ss.State(people=False)
    infection = S >> I @ ss.perday(0.5) * W.n / (W.n + 1e5)
    cholera = ss.Disease(S, I, R, W, infection=infection, recovery=ss.Transition(I, {R: 0.9, S: 0.1}, dur=ss.lognormal(5, 2, unit='days')),
                         shedding=ss.outside >> W @ ss.par(10, bounds=[5, 20], name='shed') * I.n, decay=W >> ss.dead @ ss.days(30),
                         cases=ss.outcome(0.5*infection, delay=ss.days(3)), prev=I/(S + I + R), init_n=dict(I=5))
    cholera.stratify(age=[0, 15])
    sim = ss.Sim(cholera, connectors=ss.modify(cholera.recovery, multiply=ss.by(age=[1.5, 1])), unit='day', start='2020-01-01', stop='2020-07-01', total_pop=1000, solver='gillespie', **kw)
    sim.stratify(risk={'low': 0.8, 'high': 0.2})
    sim.save('test_load.sim') # Before it has run
    sim2 = ss.load('test_load.sim').run()
    assert np.array_equal(sim2.results.cholera.cases.values, sim.run().results.cholera.cases.values) # Saving a run sim is in test_sim.py
    sc.rmpath('test_load.sim')


def test_transfer():
    sc.heading('Testing vaccination as a stratification plus a transfer')
    sims = sc.obj()
    for solver in ['ode', 'abm']:
        sir = make_sir(beta=0.2)
        sir.stratify(risk={'low': 0.8, 'high': 0.2}, vaccinated={False: 1, True: 0}, vaccination=ss.transfer(dict(vaccinated=True), rate=ss.perday(0.02), states=['S']))
        by_group = ss.modify(sir.infection, multiply=ss.by(risk=[1.0, 3.0])*ss.by(vaccinated=[1.0, 0.1]))
        sims[solver] = ss.Sim(sir, connectors=by_group, total_pop=100_000, unit='day', stop=150, solver=solver, **(dict(n_agents=20_000) if solver == 'abm' else {}), **kw).run()
    ode, abm = sims.ode.results.sir, sims.abm.results.sir
    assert ode.vaccination.cum[-1] > 50_000
    assert ode.infection.select(risk='high').sum()/20_000 > ode.infection.select(risk='low').sum()/80_000
    assert ode.infection.select(vaccinated=True).sum() < 0.1*ode.infection.sum() # The disease never mentions vaccination
    for key in ['infection', 'vaccination']:
        assert np.isclose(abm[key].cum[-1], ode[key].cum[-1], rtol=0.2), key # One run, at a daily step: a difference of order dt, plus noise

    # A transfer's dur= distribution is drawn for each person: as agents exactly, as compartments by Erlang stages; and to one of several levels, with p=
    S, I = ss.states(2)
    wane = ss.Disease(inf=S >> I @ 0*I, init=dict(I=0.0), name='wane')
    wane.stratify(rpr={False: 0, True: 1}, revert=ss.transfer(dict(rpr=False), dur=ss.lognormal(ss.years(15), ss.years(5))))
    s2 = np.log(1 + (5/15)**2)
    lognorm = stats.lognorm.cdf([8, 15], np.sqrt(s2), scale=15*np.exp(-s2/2))
    erlang = stats.gamma.cdf([8, 15], 9, scale=15/9) # 9 stages for a CV of 1/3
    for solver,expected in [('abm', lognorm), ('ode', erlang)]:
        n = ss.Sim(wane, unit='year', dt=0.5, start=0, stop=16, solver=solver, **size(solver, 5000), **kw).run().results.wane.S.by('rpr').values
        assert np.allclose(n[[16, 30], 0]/n[0].sum(), expected, atol=0.025), solver
        fp = ss.Disease(inf=S >> I @ 0*I, init=dict(I=0.0), name='fp')
        fp.stratify(method={'none': 1, 'pill': 0, 'iud': 0}, start=ss.transfer(dict(method=('none', ['pill', 'iud'])), p=[0.6, None], dur=ss.lognormal(ss.years(2), ss.years(1))))
        n = ss.Sim(fp, unit='year', dt=0.25, start=0, stop=10, solver=solver, **size(solver, 5000), **kw).run().results.fp.S.by('method').values[-1]
        assert np.allclose(n/n.sum(), [0, 0.6, 0.4], atol=0.02), solver


def test_lifetable_and_seeding():
    sc.heading('Testing deaths from a life table, and seeding a number of people on a schedule')
    ages = np.arange(101)
    table = ss.lifetable(ages, np.clip(np.gradient(1 - np.exp(-(ages/80.0)**6)) + 1e-4, 0, None)) # Most deaths in old age
    def make(strat=False):
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ 0.3*I.frac, I >> R @ ss.days(8), death=ss.death(lifetable=table), init=dict(I=0), name='sir')
        return sir.stratify(age=[0, 20, 40, 60, 70, 80, 90]) if strat else sir
    ppl = ss.People(20_000, age_data=np.array([np.arange(100), np.exp(-0.02*np.arange(100))]).T)
    sim = ss.Sim(make(), people=ppl, unit='year', dt=1, stop=3, verbose=0).init()
    assert isinstance(sim.solver, ss.solvers.abm) # Without a stratification by age, a life table needs agents
    age = np.asarray(sim.people.age.raw[sim.people.auids])
    expected = np.sum(-np.expm1(-table.hazards(ages)[np.minimum(age.astype(int), len(ages) - 1)]))
    res = sim.run().results.sir
    assert np.isclose(res.death[0], expected, rtol=0.1) and np.isclose(res.S[0] - res.S[1], res.death[0])
    assert ss.Sim(make(strat=True), unit='year', stop=3, solver='ode', verbose=0).run().results.sir.death[1] > 0 # Stratified by age, a death rate for each group

    # Three people infected every 30 days for a year, chosen from everyone susceptible, as LASER's importation
    for solver in ['ode', 'abm']:
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ ss.perday(0.05)*I.frac, I >> R @ ss.days(8), init=dict(I=0), name='sir')
        seeds = ss.Event(S >> I, when=ss.schedule(every=ss.days(30), start=ss.days(30), stop=ss.days(365)), n=3, name='importation')
        delivered = ss.Sim(sir, interventions=seeds, unit='day', stop=400, solver=solver, n_agents=10_000, total_pop=10_000, verbose=0).run().results.importation.delivered.values
        assert list(np.flatnonzero(delivered)[:3]) == [30, 60, 90] and np.allclose(delivered[delivered > 0], 3)


def test_cross_disease():
    sc.heading('Testing a rate that reads another disease\'s state, and the first state')
    def final(cross, solver):
        S, I, R = ss.states('S I R')
        syph = ss.Disease(infection=ss.Transition(S, I, rate=0.0 * I), recovery=ss.Transition(I, R, rate=ss.peryear(1e-12)), init_n=dict(I=2000), name='syph') # 10% infected throughout
        beta = 0.15*(1 + 5*syph.I.n/20_000) if cross else 0.225 # So the same
        S, I, R = ss.states('S I R')
        hiv = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, dur=10), init_n=dict(I=100), name='hiv')
        return ss.Sim([syph, hiv], total_pop=20_000, n_agents=20_000, stop=10, solver=solver, verbose=0).run().results.hiv.I[-1]
    for solver in ['ode', 'abm']:
        assert np.isclose(final(True, solver), final(False, solver), rtol=1e-9) # syph.I is syphilis's, not HIV's own I

    # The people left over start in the first state: first=, else the state named S, else the first made (which describe() says is a guess)
    R, I, S = ss.states(3)
    make = lambda **kw: ss.Disease(infection=S >> I @ 0.3*I.frac, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir', **kw)
    for solver in ['ode', 'abm']:
        assert ss.Sim(make(), total_pop=10_000, n_agents=10_000, stop=1, unit='day', solver=solver, verbose=0).run().results.sir.S[0] == 9900
        assert ss.Sim(make(default_state='R'), total_pop=10_000, n_agents=10_000, stop=1, unit='day', solver=solver, verbose=0).run().results.sir.R[0] == 9900
    B, A = ss.states('B A')
    sim = ss.Sim(ss.Disease(prog=A >> B @ 0.1, back=B >> A @ 0.1, name='d'), stop=2, verbose=0).run()
    assert sim.results.d.B[0] == 10_000 and 'the first state made' in sim.describe()


def test_seeded_stay():
    sc.heading('Testing that people infectious at the start have been so for the first step, as if infected in the step before')
    for solver in ['ode', 'binomial', 'abm']: # A seeded case with a fixed stay of one step leaves after it, as in the Reed-Frost model
        S, I, R = ss.states(3)
        rf = ss.Disease(infection=S >> I @ ss.perday(0.0)*I.n, recovery=I >> R @ ss.constant(ss.days(1)), init_n=dict(I=10), name='rf')
        sim = ss.Sim(rf, unit='day', stop=5, total_pop=100, n_agents=100, solver=solver, verbose=0).run()
        assert sim.results.rf.I[0] == 10 and sim.results.rf.I[1] == 0
    for solver in ['binomial', 'abm']: # So the seeds infect for one generation, as the people they infect do
        S, I, R = ss.states(3)
        rf = ss.Disease(infection=S >> I @ ss.perday(0.001)*I.n, recovery=I >> R @ ss.constant(ss.days(1)), init_n=dict(I=100), name='rf')
        res = ss.Sim(rf, unit='day', stop=2, total_pop=10_000, n_agents=10_000, solver=solver, verbose=0).run().results.rf
        assert res.I[1] == res.infection[0]


def test_immediate():
    sc.heading('Testing moves with no wait (dur=0, or p= with no driver): in the step that caused them, under every solver')
    for solver in ['ode', 'binomial', 'gillespie', 'abm']: # A junction holds nobody: a split, and a single destination
        S, P, A, B = ss.states('S P A B')
        d = ss.Disease(go=ss.Transition(S, P, rate=ss.perday(0.1)), triage=ss.Transition(P, {A: 0.8, B: None}, dur=0), name='d')
        res = ss.Sim(d, unit='day', stop=10, solver=solver, **size(solver), **kw).run().results.d
        assert res.P.max() == 0 and np.isclose(res.A[-1]/(res.A[-1] + res.B[-1]), 0.8, atol=0.03)
        S, P, A = ss.states('S P A')
        d = ss.Disease(go=ss.Transition(S, P, rate=ss.perday(0.1)), onward=ss.Transition(P, A, dur=0), name='d')
        res = ss.Sim(d, unit='day', stop=10, solver=solver, **size(solver), **kw).run().results.d
        assert res.P.max() == 0 and res.onward.sum() == res.go.sum()
    out = {} # With where=, as soon as the condition holds
    for solver in ['ode', 'abm']:
        I, R = ss.states('I R')
        d = ss.Disease(I, R, drop=ss.Transition(I, R, dur=0, who=ss.conditions.Cond('dim', 'level') == 'b'), default_state='I', name='d')
        d.stratify(level={'a': 1, 'b': 0}, switch=ss.transfer(dict(level='b'), rate=ss.perday(0.1)))
        out[solver] = ss.Sim(d, unit='day', stop=20, solver=solver, **size(solver), **kw).run().results.d.I[-1]/10_000
    assert np.isclose(out['ode'], np.exp(-2), rtol=0.01) and np.isclose(out['abm'], out['ode'], atol=0.02)

    # A cascade: an immediate split in the state-change row, then on again in the same step
    def make():
        S, X, E, P, A, Q, R = ss.states('S X E P A Q R')
        return ss.Disease(S, X, E, P, A, Q, R,
            infection = S >> E @ 0.4*(P + A + Q),
            onset     = ss.Transition(S, X, rate=ss.perday(0.002)),
            entry     = ss.Transition(X, E, dur=2),
            split_p   = ss.Transition(E, P, p=0.3), # No driver: split on entry, at once
            split_a   = ss.Transition(E, A, p=0.7),
            onward    = ss.Transition(P, Q, p=1.0),
            rec_q     = Q >> R @ ss.days(5),
            rec_a     = A >> R @ ss.days(5),
            init_n=dict(E=50), name='d')
    for solver in ['ode', 'abm', 'binomial', 'gillespie']:
        res = ss.Sim(make(), solver=solver, unit='day', stop=60, **size(solver, 5000), **kw).run().results.d
        into = res.infection[1:] + res.entry[1:]
        split = res.split_p + res.split_a
        if solver in ['ode', 'abm']:
            assert res.E.max() == 0 and res.P.max() == 0, solver # Nobody is ever counted in a state with no wait
            assert np.allclose(split[1:], into) and np.isclose(res.onward.sum(), 0.3*split.sum(), rtol=0.1), solver
        else: # Each person once, through one branch, and on from P only
            assert np.isclose(split.sum(), into.sum(), rtol=0.03) and res.onward.sum() == res.split_p.sum(), solver


def test_dwells():
    sc.heading('Testing dwell times: following a person between levels, as expressions, scheduled at entry, and a minimum stay beside another dwell')
    out = {} # Recovery is ten times slower in b, and people move there at once: the dwell drawn in a follows them
    for solver in ['ode', 'abm']:
        I, R = ss.states('I R')
        d = ss.Disease(rec=I >> R @ ss.days(20), default_state='I', name='d')
        d.stratify(level={'a': 1, 'b': 0}, switch=ss.transfer(dict(level='b'), rate=ss.perday(0.5)))
        out[solver] = ss.Sim(d, connectors=ss.modify(d.rec, multiply=ss.by(level=[1, 0.1])), unit='day', dt=0.25, stop=20, solver=solver, verbose=0, **size(solver, 20_000)).run().results.d.I[-1]/20_000
    assert np.isclose(out['abm'], out['ode'], atol=0.03)

    # A distribution times a factor scales each draw, so keeps its shape
    def still_in(mode, factor=1.0, years=(1, 5, 20, 50)):
        S, I, R = ss.states('S I R')
        k = ss.attr(factor)
        d0 = ss.lognormal(ss.years(5), ss.years(20))
        dur = dict(plain=d0, attr=d0*k)[mode]
        d = ss.Disease(S, I, R, rec=ss.Transition(I, R, dur=dur), k=k, init_n=dict(I=20_000), name='d')
        n = ss.Sim(d, n_agents=20_000, start=2000, stop=2060, dt=0.25, verbose=0).run().results.d.I
        return np.array([n[int(y/0.25)]/20_000 for y in years])
    assert np.allclose(still_in('attr'), still_in('plain'), atol=0.02)

    # Exits are scheduled at entry: p= is read then, and nobody in a state has no scheduled exit
    early = lambda sim: 1.0*(sim.ti < 5)
    def check(sim): # An HIV-like hook after the step
        d = sim.diseases.d
        for name in ['A', 'R']:
            assert not np.isnan(d[f'ti_exit_{name}'][d[name].uids]).any(), f'People in {name} have no scheduled exit at step {sim.ti}'
    S, A, H, R = ss.states('S A H R')
    d = ss.Disease(infection=ss.Transition(S, A, rate=0.4 * A), severe=ss.Transition(A, H, p=early, dur=3), mild=ss.Transition(A, R, p=lambda sim: 1 - early(sim), dur=3),
                   relapse=ss.Transition(R, A, dur=20), check=check, init_prev=0.01, name='d')
    res = ss.Sim(d, n_agents=5000, unit='day', stop=40, **kw).run().results.d
    assert res.severe.sum() == res.A[0] + res.infection[:5].sum() + res.relapse[:5].sum()

    # Separate p= transitions split on entry; one transition with several destinations splits when it is taken
    late = lambda sim: 1.0*(sim.ti >= 10)
    for solver in ['ode', 'abm']:
        res = []
        for shape in ['entry', 'exit']:
            I, H, R = ss.states('I H R')
            trs = dict(severe=ss.Transition(I, H, p=late, dur=20), mild=ss.Transition(I, R, p=lambda sim: 1 - late(sim), dur=20)) if shape == 'entry' else dict(course=ss.Transition(I, {H: late, R: None}, dur=20))
            res.append(ss.Sim(ss.Disease(I, H, R, **trs, default_state='I', name='d'), solver=solver, stop=100, **size(solver), **kw).run().results.d)
        assert res[0].H[-1] == 0 and np.isclose(res[1].H[-1]/10_000, np.exp(-10/20), atol=0.03), solver

    # A minimum stay on one exit beside the dwell time of another, as compartments: the dwell runs from entry
    def make(dur, after=True):
        A, E, I, D = ss.states('A E I D')
        d = ss.Disease(A, E, I, D, enter=A >> E @ ss.perday(0.3), onset=ss.Transition(E, I, dur=dur), die=ss.Transition(E, D, rate=ss.perday(0.0), min_stay=ss.days(2) if after else None),
                       init=dict(A=1.0), default_state='A', name='d')
        return ss.Sim(d, unit='day', stop=20, total_pop=1e5, solver='ode', verbose=0).run().results.d.I.values
    for dur in [ss.normal(ss.days(5), ss.days(1)), ss.choice(a=ss.days([3, 4, 5]), p=[0.2, 0.5, 0.3])]: # As stages in each slot of the stay, and as a conveyor of whole days
        assert np.allclose(make(dur), make(dur, after=False), rtol=1e-6, atol=0.01)

    # The time people spent in a state (E.stays): the declared mean, kept without keep_times, and not changed by going back to a state it led to
    S, E, I, R = ss.states('S E I R')
    m = ss.Sim(ss.Disease(S >> E @ 1.9*I, E >> I @ ss.days(10), I >> R @ ss.days(8), init_n=dict(I=10), name='seir'), n_agents=5000, dur=ss.days(200), rand_seed=1, verbose=0).run().diseases.seir
    assert isinstance(m.E.stays, ss.days) and len(m.E.stays) > 4000 and np.isclose(np.mean(m.E.stays).value, 10, atol=0.5) # The seeds, who never entered E, are not counted
    assert np.isclose(np.mean(m.I.stays).value, 8, atol=0.5)


def test_agent_steps():
    sc.heading('Testing the agent engine: half a step of rates at the first draw, whole steps, the compiled moves, and values fixed for life')
    S, I, R = ss.states(3)
    sir = ss.Disease(S, I, R, infection=S >> I @ ss.perday(0.25)*I, recovery=I >> R @ ss.perday(1/8), init_prev=0.001, name='sir') # R0 = 2
    attack = lambda solver: 1 - ss.Sim(sir, solver=solver, n_agents=20_000, unit='day', stop=250, dt=ss.days(4), rand_seed=1, verbose=0).run().results.sir.S[-1]/20_000
    assert np.isclose(attack('abm'), 0.797, atol=0.02) # At a step of half the infectious period, R0 holds (0.61 with a whole step at the first draw)
    assert 0.5 < attack(ss.solvers.abm(step='wholestep')) < 0.75 # A whole step of recovery at the first draw, as written as code with ss.bernoulli()

    # With step='wholestep', rates act on those in the state at the start of the step only; a module's own way of running overrides the sim's
    def make(**kwargs):
        A, B, C = ss.states('A B C')
        return ss.Disease(A, B, C, tick=A >> B @ ss.constant(ss.days(3)), leave=B >> C @ ss.perday(0.5), default_state='A', name='d', **kwargs) # Everyone reaches B in step 3
    res = {}
    for step in ['midstep', 'wholestep']:
        sim = ss.Sim(make(), solver=ss.solvers.abm(step=step), n_agents=10_000, unit='day', stop=10, verbose=0).run()
        assert not sim.diseases.d.engine.uncompiled and sim.diseases.d.engine.wholestep == (step == 'wholestep')
        res[step] = sim.results.d
    assert res['midstep'].tick[3] == 10_000 and np.isclose(res['midstep'].leave[3]/10_000, 1 - np.exp(-0.25), atol=0.02) and res['wholestep'].leave[3] == 0
    assert np.isclose(res['wholestep'].leave[4]/10_000, 1 - np.exp(-0.5), atol=0.02)
    wholestep = lambda sim_solver, **kw: ss.Sim(make(**kw), solver=sim_solver, n_agents=500, unit='day', stop=5, verbose=0).init().diseases.d.engine.wholestep
    assert wholestep('abm', solver=ss.solvers.abm(step='wholestep')) and not wholestep(ss.solvers.abm(step='wholestep'), solver='abm')

    # The compiled moves make the draws the agent engine's own moves make
    def make(seed):
        S, E, I, H, R, F = ss.states('S E I H R F')
        infection = S >> E @ ss.perday(0.3)*(I + H)
        d = ss.Disease(S, E, I, H, R, F, infection=infection,
            onset   = ss.Transition(E, {I: 0.8, R: None}, dur=ss.lognormal(ss.days(3), ss.days(1))), # A split after a dwell time
            severe  = ss.Transition(I, H, p=0.2, dur=ss.weibull(shape=2, scale=ss.days(4))), # Branches chosen on entry
            mild    = ss.Transition(I, {R: 0.9, ss.stay: None}, dur=ss.uniform(ss.days(3), ss.days(9)), p=None),
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
                assert np.array_equal(v.raw, own[k].raw, equal_nan=True), k

    # A function marked with ss.lifelong() is worked out once for each person, and gives the same run as unmarked
    female = lambda sim: np.asarray(sim.people.female.raw, dtype=bool)
    frac = lambda sim, uids: np.where(sim.people.female[uids], 0.8, 0.2)
    def make(mark):
        f = ss.lifelong if mark else (lambda func: func)
        S, I, H, R = ss.states('S I H R')
        return ss.Disease(S, I, H, R, infection=ss.Transition(S, I, rate=0.3*I), course=ss.Transition(I, {H: f(frac), R: None}, dur=ss.days(8)),
                          recover=ss.Transition(H, R, dur=ss.days(10), who=f(female)), init_prev=0.02, name='d')
    sims = [ss.Sim(make(mark), n_agents=5000, unit='day', stop=60, demographics=[ss.Births(birth_rate=ss.peryear(0.03))], rand_seed=3, verbose=0) for mark in [False, True]]
    assert same(*sims)


def test_held_rate():
    sc.heading('Testing a rate held in a parameter (ss.par(beta*I)), and a function that reads a parameter')
    def make(held, solver='ode', **kwargs):
        S, I, R = ss.states('S I R')
        rate = ss.perday(0.3)*I
        sir = ss.Disease(infection=S >> I @ (ss.par(rate, name='force') if held else rate), recovery=I >> R @ ss.days(8), init=dict(I=0.01), name='sir')
        return ss.Sim(sir, unit='day', stop=60, total_pop=10_000, solver=solver, verbose=0, **kwargs)
    for solver,kwargs in [('ode', {}), ('abm', dict(n_agents=2000, networks=ss.RandomNet(n_contacts=4)))]:
        held, plain = make(True, solver, **kwargs).run(), make(False, solver, **kwargs).run()
        assert np.array_equal(held.results.sir.I.values, plain.results.sir.I.values), solver
    assert held.diseases.sir.graph.transitions.infection.contact and np.isclose(make(True).init().diseases.sir.R0(), 2.4)

    # A function that reads a parameter sees a change to it
    def initiation(sim, uids):
        return ss.peryear(1/(1 + np.exp(-sim.diseases.meth.pars.b0))*np.ones(len(uids)))
    starts = []
    for change in [False, True]:
        none, on = ss.states('none on')
        m = ss.Disease(start=ss.Transition(none, on, rate=initiation), stop=ss.Transition(on, none, rate=ss.peryear(0.5)), init=dict(none=1.0), b0=ss.par(-1.0), default_state='none', name='meth')
        intv = ss.modify('meth.pars.b0', set=-20.0, when=ss.schedule(start=2005)) if change else None
        starts.append(ss.Sim(m, interventions=intv, start=2000, stop=2010, dt=1/12, n_agents=2000, verbose=0).run().results.meth.start[61:].sum())
    assert starts[0] > 100 and starts[1] == 0


def test_value():
    sc.heading('Testing that a parameter has one value for code: arithmetic on self.pars.x, declared or written as code')
    sir = ss.SIR(beta=0.3)
    assert isinstance(1. - sir.pars.beta, ss.expressions.BinOp), 'Before the sim is initialized, arithmetic on a parameter is part of the model'
    p = ss.Sim(sir, n_agents=200, verbose=0).init().diseases.sir.pars.beta
    for x in [1., np.array(1.), np.ones(2)]: # After, it is a value, whatever the shape of the other operand
        assert np.all(x - p == 0.7) and not isinstance(x - p, ss.expressions.Expr)

    class Reader(ss.Module): # Read from code: arithmetic applies a modifier's change, and float() is the value the parameter holds
        def step(self):
            self.read = [1*self.sim.diseases.sir.pars.beta, float(self.sim.diseases.sir.pars.beta), 2*self.pars.death_ts]
        def __init__(self):
            super().__init__()
            self.define_pars(death_ts=ss.timeseries({'2000-01-01':ss.peryear(1), '2002-01-01':ss.peryear(2)}))
    sim = ss.Sim(ss.SIR(beta=0.3), Reader(), interventions=ss.modify('sir.pars.beta', set=0.1, when=ss.schedule(start='2001-01-01')), start='2000-01-01', stop='2003-01-01', unit='day', dt=ss.years(1), n_agents=200, verbose=0).run()
    beta, raw, death2 = [m for m in sim.modules if isinstance(m, Reader)][0].read
    assert np.isclose(beta, 0.1) and raw == 0.3 and isinstance(death2, ss.peryear) and death2.value == 4 # A time series of typed values gives typed values


def test_init_prev_forms():
    sc.heading('Testing init_prev as a function and by people\'s properties: for each agent, or for each group')
    def make(**kwargs):
        S, Ic, In, R = ss.states('S Ic In R')
        return ss.Disease(infection=ss.Transition(S, {Ic: 0.3, In: 0.7}, rate=0.1*(Ic + In)), clear_c=Ic >> R @ ss.years(1), clear_n=In >> R @ ss.years(1), name='d', **kwargs)
    by_age = ss.by(dict(age=[0, 20], value=[0.0, 0.2])) # By each person's own age
    for kwargs in [dict(init_prev=by_age), dict(init=dict(Ic=by_age))]: # A function of the sim is in test_dimensions.py
        sim = ss.Sim(make(**kwargs), n_agents=20_000, stop=2001, **kw).init()
        uids = sim.people.auids
        old = sim.people.age.raw[uids] >= 20
        seeded = sim.diseases.d.Ic.raw[uids] | sim.diseases.d.In.raw[uids]
        assert isinstance(sim.solver, ss.solvers.abm) and seeded[~old].sum() == 0 and np.isclose(seeded[old].mean(), 0.2, atol=0.01)
    by_risk = ss.Sim(make(init_prev=ss.by(risk=[0.0, 0.2])).stratify(risk=['low', 'high']), solver='ode', total_pop=1000, stop=2001, **kw).init() # By a dimension, for each group
    assert np.isclose(by_risk.diseases.d.engine.count(by_risk.diseases.d.graph.people_states[1:3]), 100)


def test_branch_ends():
    sc.heading('Testing ss.dead as a branch: a share die when the transition is taken, by group, adjusted, and modified')
    for solver in ['abm', 'ode', 'binomial']:
        S, I, R = ss.states('S I R')
        d = ss.Disease(S, I, R, infection=ss.Transition(S, I, rate=0.0), death=ss.Transition(I, {R: 0.9, ss.dead: 0.1}, dur=ss.days(10)), init=dict(S=0.5, I=0.5), name='d') # Those in I at the start each take it once
        sim = ss.Sim(d, unit='day', stop=200, solver=solver, verbose=0, **size(solver, 20_000)).run()
        r = sim.results.d
        dead = r['death.dead'].values.sum()
        assert np.isclose(dead/r.I[0], 0.1, rtol=0.1) and np.isclose(r.R[-1]/r.I[0], 0.9, rtol=0.02), solver
        assert np.isclose(r.S[-1] + r.I[-1] + r.R[-1] + dead, r.S[0] + r.I[0]), solver
        if solver == 'abm':
            assert sim.results.death.values.sum() == dead # Counted as deaths, as a transition to ss.dead is

    # By group, adjusted by group, and modified, with the other branches sharing what is left
    def make(dead=0.02, by_age=False):
        S, I, R, H = ss.states('S I R H')
        dis = ss.Disease(infection=S >> I @ ss.perday(0.3)*I, outcome=I >> {H: 0.1, ss.dead: dead, R: None} @ ss.days(8), init_n=dict(I=10), name='sir')
        return dis.stratify(age={0: 0.25, 18: 0.6, 65: 0.15}) if by_age else dis
    run = lambda dis, **kw: ss.Sim(dis, total_pop=10_000, dur=ss.days(200), verbose=0, solver='ode', **kw).run().results
    frac = lambda r: r.death.sum()/(r.death.sum() + r.sir.H[-1] + r.sir.R[-1])
    assert np.isclose(frac(run(make())), 0.02)
    assert np.isclose(frac(run(make(dead=ss.by(age=[0.001, 0.01, 0.1]), by_age=True))), 0.25*0.001 + 0.6*0.01 + 0.15*0.1)
    assert np.isclose(frac(run(make(by_age=True), connectors=ss.modify('sir.outcome.dead', multiply=ss.by(age=[0.1, 1, 5])))), 0.02*(0.25*0.1 + 0.6 + 0.15*5))
    assert np.isclose(frac(run(make(), connectors=ss.modify('sir.outcome.H', multiply=3))), 0.02*0.7/0.9) # So deaths do not vanish

    S, I, R, F = ss.states('S I R F') # A split to one destination is a plain transition, so several out of one state compete as rates
    d = ss.Disease(S >> I @ 0.3*I, a=ss.Transition(I, [F], p=[1.0], rate=0.1), b=ss.Transition(I, [R], p=[1.0], rate=0.05), name='d')
    assert d.a.p is None and len(ss.Sim(d, n_agents=100, verbose=0).init().diseases.d.graph.exits.I.hazards) == 2


def test_read_each_step():
    sc.heading('Testing what a run taken whole, and one stepped, read at each step: other diseases, values in time, and alive')
    class Step(ss.Intervention): # Code each step, so the run is stepped
        def step(self):
            pass
    def sir(name='sir', rate=lambda I: 0.3*I, results=lambda I: {}):
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=ss.Transition(S, I, rate=rate(I)), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name=name, **results(I))
    pars = dict(unit='day', stop=200, total_pop=50_000, solver='ode', verbose=0)
    def pair():
        a = sir('a')
        return [a, sir('b', rate=lambda I: 0.3*I*(1 + 5*a.I.frac))]
    whole, stepped = [ss.Sim(pair(), interventions=ints, **pars).run().results.b.I for ints in [None, Step(name='step')]] # A count of another disease is not frozen
    assert np.allclose(whole, stepped) and np.isclose(np.max(whole), 25_059, rtol=1e-3)
    b = ss.par(ss.seasonal(ss.perday(0.3), 0.5), name='b')
    make = lambda: sir(rate=lambda I: b*I, results=lambda I: dict(foi=b*I))
    pars = sc.mergedicts(pars, dt=30, stop=720)
    whole, stepped = [ss.Sim(make(), interventions=ints, **pars).run().results.sir for ints in [None, Step(name='step')]]
    assert np.allclose(whole.foi, stepped.foi)
    assert np.allclose((np.array(whole.foi)/np.array(whole.I))[:3], [0.45, 0.43044, 0.37686], atol=1e-4)

    # Only the states marked infected count when some are marked, and alive exists during a compartmental run
    S, I, R = ss.states('S I R')
    E = ss.State(infected=True) # I, the source of infection, is not marked
    d = ss.Disease(S, E, I, R, infection=S >> E @ ss.perday(0.5)*I, onset=E >> I @ ss.days(3), recovery=I >> ss.dead @ ss.days(7), init_n=dict(I=10), name='seir')
    seen = {}
    sim = ss.Sim(d, connectors=ss.modify('seir.infection', multiply=lambda sim: seen.update({sim.ti: sim.results.alive[sim.ti]}) or 1.0), total_pop=10_000, dur=ss.days(60), verbose=0, solver='ode').run()
    tis = sorted(seen)
    assert np.allclose([seen[ti] for ti in tis], sim.results.alive[tis]) and seen[tis[-1]] < seen[0]
    assert np.isclose(sim.diseases.seir.R0(), 3.5) and np.allclose(sim.results.seir.infected, sim.results.seir.E)


def test_noise():
    sc.heading('Testing noise on a rate: mean 1 and variance sigma²/h, none at sigma = 0, a parameter, and refused where it cannot be drawn')
    rng = np.random.default_rng(1)
    for sigma, h in [(0.0878, 1/365.25), (0.3, 0.5)]: # A shape below 1 and above it
        x = np.array([ss.solvers.numba.compartmental.noise(rng, sigma, h) for _ in range(20_000)])
        var = sigma**2/h
        assert abs(x.mean() - 1) < 4*np.sqrt(var/len(x)) and abs(x.var()/var - 1) < 0.15, (x.mean(), x.var(), var)
    def make(noise=None, solver='binomial'):
        S, I, R = ss.states('S I R')
        extra = {} if noise is None else dict(noise=noise)
        d = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3)*I, **extra), recovery=I >> R @ ss.perday(0.1), init_n=dict(I=20), name='sir')
        return ss.Sim(d, solver=solver, total_pop=20_000, unit='day', start=0, stop=120, verbose=0)
    for solver in ['binomial', 'tau', 'abm']: # Drawn for each substep, each leap, and as agents once a step for each group
        base = make(solver=solver).run().results.sir.I
        assert np.array_equal(base, make(noise=0.0, solver=solver).run().results.sir.I)
        assert not np.array_equal(base, make(noise=ss.par(0.5), solver=solver).run().results.sir.I)
    assert np.array_equal(make(noise=0.2, solver='ode').run().results.sir.I, make(solver='ode').run().results.sir.I) # Left out of the ODE
    sim = make(noise=0.2).init()
    assert sim.diseases.sir.pars.noise_infection.value == 0.2 # Named after its slot, so it can be fitted
    with pytest.raises(ss.CapabilityError):
        make(noise=0.2, solver='gillespie').run()


def test_approximations():
    sc.heading('Testing that each approximation follows ss.options.approximations: a note with "note", kept quietly with "allow", an error with "error" (and agents under "auto")')
    def noisy():
        S, I, R = ss.states('S I R')
        return dict(diseases=ss.Disease(ss.Transition(S, I, rate=0.3*I, noise=0.1), I >> R @ ss.days(10), init_prev=0.01, name='sir'))
    def boxed():
        return dict(diseases=ss.SIR(), interventions=ss.vaccinate(ss.Vaccine(efficacy=0.8, waning=ss.boxcar(plateau=ss.days(30))), coverage=0.5, start=ss.days(1)))
    def delayed():
        return dict(diseases=make_sir(dur=ss.constant(ss.days(2.5))))
    for make, word in [(noisy, 'noise'), (boxed, 'plateau'), (delayed, 'agents round')]:
        def sim(solver, approximations, run=True):
            with ss.options.context(approximations=approximations):
                sim = ss.Sim(**make(), total_pop=1000, stop=ss.days(10), unit='day', solver=solver, verbose=0)
                return sim.run() if run else sim.init()
        for approximations in ['note', 'allow']:
            assert any(word in note for note in sim('ode', approximations).notes), word
        with pytest.raises(ss.CapabilityError, match='approximations="error"'):
            sim('ode', 'error')
        assert isinstance(sim('auto', 'error', run=False).solver, ss.solvers.abm)


def test_treatment():
    sc.heading('Testing transitions with a rate of 0, moved by treatments, and the forms of cure=')
    def make_sis(cure=lambda S, I: I >> S @ 0):
        S, I = ss.states(2)
        return ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> S @ ss.days(20), cure=cure(S, I), init_prev=0.05, name='sis')
    sis = make_sis()
    assert sis.cure.undriven
    assert not make_sis(lambda S, I: I >> S @ ss.par(0, name='cure_rate')).cure.undriven # A parameter of 0 (e.g. fitted) is a rate
    with warnings.catch_warnings():
        warnings.simplefilter('error') # Nothing names it, which is neither a warning nor a note: it exists for products to drive
        ss.Sim(sis, stop=10, unit='day', **kw).init()
    with pytest.raises(ValueError): # With no rate at all, an error
        make_sis(lambda S, I: I >> S)
    for solver in ['ode', 'abm']:
        tx = ss.treat(ss.Treatment(cure={'cure': 0.9}), when=ss.days(20), coverage=0.3)
        res = ss.Sim(make_sis(), interventions=tx, solver=solver, stop=40, unit='day', **size(solver), **kw).run().results
        assert res.sis.cure.sum() > 0 and np.isclose(res.sis.cure.sum(), 0.9*res.tx.delivered.sum(), rtol=0.05), solver

    # cure= and moves= as a number, by state, by transition, by name, and a move written with the states
    forms = dict(
        number = lambda sis: dict(cure=0.9),
        state  = lambda sis: dict(cure=dict(I=0.9)),
        trans  = lambda sis: dict(cure={sis.cure: 0.9}),
        name   = lambda sis: dict(cure=dict(cure=0.9)),
        move   = lambda sis: dict(cure={sis.I >> sis.S: 0.9}),
        moves  = lambda sis: dict(moves=sis.cure, cure=0.9), # Not every move is a cure, so moves= names them too
        movesdict = lambda sis: dict(moves={'cure': 0.9}),
    )
    cured = {}
    for key,form in forms.items():
        sis = make_sis()
        tx = ss.treat(ss.Treatment(**form(sis)), when=ss.days(20), name='tx')
        cured[key] = ss.Sim(sis, interventions=tx, solver='ode', stop=40, unit='day', **kw).run().results.tx.cured.sum()
    assert np.allclose(list(cured.values()), cured['number'], rtol=1e-6), cured
    S, I = ss.states(2)
    with pytest.raises(TypeError): # A transition alone is not a probability
        ss.Treatment(cure=I >> S)


if __name__ == '__main__':
    for name,test in list(globals().items()):
        if name.startswith('test_'):
            test()
