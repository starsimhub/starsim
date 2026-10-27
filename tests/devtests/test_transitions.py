"""
Test models written as declared states and transitions, and the solvers that run them
"""

# %% Imports and settings
import re
import sys
import types
import time
import _thread
import functools
import itertools
import pytest
import numpy as np
import pandas as pd
import scipy.stats as stats
import sciris as sc
import starsim as ss
import starsim.fit
import starsim.library as ssl

kw = dict(verbose=0)


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
        shedding  = ss.null >> W @ xi * I.n,
        decay     = W >> ss.null @ ss.days(30),
        init      = dict(I=0.001),
    )
    return cholera


@sc.timer()
def test_counts():
    sc.heading('Testing states that are not people')
    cholera = make_cholera()
    graph = cholera.graph
    assert [s.name for s in graph.count_states] == ['W'] # Declared people=False
    sim = ss.Sim(cholera, total_pop=100_000, unit='day', stop=200, **kw).run()
    assert sim.solver == 'ode' # A count never makes a model agent-based
    res = sim.results.cholera
    assert np.allclose(res.S + res.I + res.R, 100_000) # W is not part of the population
    assert res.W.max() > 1e6

    # With agents, the people are agents and the water stays a count
    sim2 = ss.Sim(cholera, total_pop=100_000, n_agents=5_000, unit='day', stop=200, **kw).run()
    res2 = sim2.results.cholera
    assert sim2.solver == 'abm'
    assert np.isclose(res2.W.max(), res.W.max(), rtol=0.2)

    # A state that no transition touches is people, as every state not declared a count is: it is where people are put, e.g. by init=
    S, I, R, Q = ss.states('S I R Q')
    sir = ss.Disease(S, I, R, Q, infection=S >> I @ ss.perday(0.3)*I, recovery=I >> R @ ss.days(10), init=dict(I=0.01, Q=0.1), name='sir')
    assert [s.name for s in sir.graph.people_states] == ['S', 'I', 'R', 'Q']
    for solver in ['ode', 'abm']:
        sim = ss.Sim(sir, solver=solver, total_pop=10_000, unit='day', stop=50, **kw).run()
        res = sim.results.sir
        assert np.allclose(res.S + res.I + res.R + res.Q, 10_000, rtol=0.01) and np.isclose(res.Q[-1], 1000, rtol=0.1)
        assert not any('treated as a count' in note for note in sim.notes)

    # A fixed delay with a competing exit (a mosquito's incubation, and its mortality), which acts over the whole delay in the ODE as in the chain binomial: exactly 5,020 infectious (it was 5,279, half a step of mortality short)
    def eip():
        Sh, Ih = ss.states(2)
        Sv, Ev, Iv = ss.states(3, people=False)
        return ss.Disease(infection=Sh >> Ih @ ss.perday(1e-9)*Iv.n/(Sh + Ih).n, emergence=ss.null >> Sv @ ss.perday(5000), biting=Sv >> Ev @ ss.perday(0.05),
                          incubation=Ev >> Iv @ ss.constant(ss.days(12)), mortality=Sv + Ev + Iv >> ss.null @ ss.perday(0.1), init=dict(Sh=1.0, Sv=33_333), first='Sh', name='eip')
    for solver in ['ode', 'binomial']:
        Iv = ss.Sim(eip(), solver=solver, total_pop=2000, unit='day', stop=300, **kw).run().results.eip.Iv[-1]
        assert np.isclose(Iv, 5000*0.05/0.15*np.exp(-1.2)/0.1, rtol=0.002), f'{solver}: {Iv:.0f} infectious mosquitoes, not 5,020'

    # Mosquitoes are counts when declared so: ss.states() takes people=
    def vector():
        Sh, Ih = ss.states(2)
        Sv, Iv = ss.states(2, people=False)
        return ss.Disease(infection=Sh >> Ih @ ss.perday(0.1)*Iv.n/(Sh + Ih).n, recovery=Ih >> Sh @ ss.days(50), emergence=ss.null >> Sv @ ss.perday(100),
                          biting=Sv >> Iv @ ss.perday(0.1)*Ih.n/(Sh + Ih).n, mortality=Sv + Iv >> ss.null @ ss.days(10), init_n=dict(Sv=3000), first='Sh', name='vector')
    assert [s.name for s in vector().graph.people_states] == ['Sh', 'Ih']

    # As agents, counts take substeps (at a monthly step they died out) and the modifiers of their transitions (which were ignored)
    for dt,mult in [(30, 1.0), (1, 2.0)]:
        sim = ss.Sim(vector(), interventions=ss.modify('vector.mortality', multiply=mult), n_agents=1000, unit='day', stop=360, dt=dt, **kw).run()
        assert np.isclose(sim.results.vector.Sv[-1] + sim.results.vector.Iv[-1], 1000/mult, rtol=0.01) # Emergence over mortality

    # A bare count in a rate is an error
    with pytest.raises(ValueError, match='there is no contact with a count'):
        S, I = ss.states(2)
        W = ss.State(people=False)
        ss.Disease(infection = S >> I @ 0.5 * W, shedding = ss.null >> W @ 10 * I.n)

    # An inflow split between states of people (FPsim review: primary infertility, reached only from births)
    S, X = ss.states('S X')
    d = ss.Disease(S, X, births=ss.Transition(ss.null, [S, X], rate=ss.peryear(100), p=[0.9, 0.1]), name='d')
    assert X.people
    assert np.isclose(ss.Sim(d, n_agents=1000, stop=5, **kw).run().results.d.X[-1], 50, atol=25)
    with pytest.raises(ValueError, match='people=False'): # Unless one is declared not to be
        S, X = ss.states('S X')
        X.people = False
        ss.Disease(S, X, births=ss.Transition(ss.null, [S, X], rate=ss.peryear(100), p=[0.9, 0.1]))

    # A count that is not people is a real number under every solver: a reservoir shed into by a few infected keeps them infected, as in SimInf's SISe
    def pressure():
        S, I = ss.states(2)
        phi = ss.State(people=False, name='phi')
        return ss.Disease(S, I, phi, infection=S >> I @ ss.perday(0.1)*phi.n, recovery=I >> S @ ss.perday(0.05), shed=ss.null >> phi @ ss.perday(1)*I.n/(S+I).n,
                          decay=phi >> ss.null @ ss.perday(0.1), init_n=dict(I=10), name='pressure')
    ode = ss.Sim(pressure(), total_pop=100, unit='day', stop=200, **kw).run().results.pressure.I[-1]
    for solver,agents in [('binomial', {}), ('gillespie', {}), ('abm', dict(n_agents=100))]:
        final = np.mean([ss.Sim(pressure(), total_pop=100, unit='day', stop=200, solver=solver, rand_seed=seed, **agents, **kw).run().results.pressure.I[-1] for seed in range(5)])
        assert np.isclose(final, ode, rtol=0.1), f'Expected {solver} to keep the infection going as the ODE does'

    # NumPy functions of counts in a rate: the same in the kernels and as agents
    def capped():
        S, I, R = ss.states(3)
        return ss.Disease(infection = S >> I @ np.clip(0.5 * I.frac, 0, 0.02) * np.exp(0 * I.n), recovery = I >> R @ ss.days(10), init=dict(I=0.01), name='capped')
    sim = ss.Sim(capped(), unit='day', stop=150, **kw).run()
    kernel = sim.results.capped.I.values
    agents = ss.Sim(capped(), unit='day', stop=150, n_agents=10_000, **kw).run().results.capped.I.values
    assert np.allclose(agents[60], kernel[60], rtol=0.1)
    assert 'minimum(maximum(0.5 * I.frac, 0), 0.02)' in ss.expressions.format_expr(sim.diseases.capped.graph.transitions[0].rate, exact=True)

    # A transition for some people only
    ppl = ss.People(1)
    S, I, R = ss.states('S I R')
    older = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, rate=ss.perday(0.1)),
                       vaccination=ss.Transition(S, R, rate=ss.perday(0.05), where=ppl.age >= 50), init_prev=0.01, name='older')
    older.stratify(age=[0, 50])
    for solver,agents in [('ode', {}), ('abm', dict(n_agents=5000))]:
        vaccinated = ss.Sim(older, unit='day', stop=50, solver=solver, **agents, **kw).run().results.older.vaccination.by('age').values.sum(axis=0)
        assert vaccinated[0] == 0 and vaccinated[1] > 500

    # A function of the sim that uses the model's own states (I.n) is the same as the expression written out
    def closed(fn):
        S, I, R = ss.states(3)
        rate = (lambda sim: np.clip(I.n/1000, 0, 1)*0.5) if fn else np.clip(I.n/1000, 0, 1)*0.5
        return ss.Disease(S, I, R, infection=S >> I @ 0.3*I, recovery=I >> R @ rate, init_prev=0.01, name='closed')
    assert np.allclose(*[ss.Sim(closed(fn), unit='day', stop=60, **kw).run().results.closed.I for fn in [True, False]])

    # where= can be a state of another module, and a death can have several sources
    J, A, X, Y = ss.states('J A X Y')
    growth = ss.Disease(maturity=J >> A @ ss.perday(0.1), first='J', name='growth')
    adults = ss.Disease(shift=ss.Transition(X, Y, rate=ss.perday(1.0), where=growth.A), first='X', name='adults')
    assert ss.Sim([growth, adults], n_agents=100, unit='day', stop=5, **kw).run().results.adults.shift.sum() > 0
    assert [s.name for s in ss.death([J, A], rate=0.1).src] == ['J', 'A']
    A, B, C = ss.states('A B C')
    d = ss.Disease(A, B, C, tr=ss.Transition(A + B, C, rate=ss.peryear(0.1)), name='d') # A source that is a sum of states is several sources, as in A + B >> C
    assert [s.name for s in d.graph.transitions.tr.src] == ['A', 'B']

    # A value set each step that is not a flow, as odin's update() of a variable (the largest prevalence so far, which slows the spread): a module written as code, read by a function in the rate
    class Track(ss.Connector):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.peak = 0.0
            self.peaks = []
        def step(self):
            d = self.sim.diseases.behaviour
            self.peak = max(self.peak, d.I.n/self.sim.people.alive.sum())
            self.peaks.append(self.peak)
    slowing = lambda sim: np.exp(-5*sim.connectors.track.peak)
    S, I, R = ss.states(3)
    behaviour = ss.Disease(infection=S >> I @ ss.perday(0.3)*I*slowing, recovery=I >> R @ ss.days(10), init_n=dict(I=10), name='behaviour')
    sim = ss.Sim(behaviour, connectors=Track(name='track'), unit='day', stop=200, n_agents=10_000, **kw).run()
    frac = sim.results.behaviour.I.values/10_000
    assert np.allclose(sim.connectors.track.peaks, np.maximum.accumulate(frac)[:-1]) # Set from the start of each step
    assert frac.max() < 0.15 # Without the slowing, 30% are infectious at the peak
    ode = ss.Sim(behaviour, connectors=Track(name='track'), unit='day', stop=200, total_pop=10_000, solver='ode', **kw).run() # The same code, as compartments: the number alive is the number of people, not of groups
    assert ode.people.alive.sum() == 10_000 and np.isclose(ode.connectors.track.peak, frac.max(), rtol=0.2)

    # A comparison is 1 or 0, so a rate can stop at a capacity
    S, I, Q, R = ss.states(4)
    quarantine = ss.Disease(S, I, Q, R, infection=S >> I @ 0.3*I, isolate=I >> Q @ ss.perday(0.5)*(Q.n < 30), recovery=I >> R @ ss.days(10), release=Q >> R @ ss.days(5), init_prev=0.01, name='quarantine')
    assert 29 < ss.Sim(quarantine, unit='day', dt=ss.days(0.01), stop=60, **kw).run().results.quarantine.Q.max() < 30 + 0.5*10_000*0.01 # Up to one step's arrivals over it
    S, I, Q = ss.states('S I Q') # Or as where=
    d = ss.Disease(S, I, Q, infection=S >> I @ ss.peryear(2)*I, q=ss.Transition(I, Q, rate=ss.peryear(5), where=Q.n < 20), init_prev=0.1, name='d')
    assert ss.Sim(d, n_agents=1000, start=2000, stop=2003, dt=ss.years(0.25), **kw).run().results.d.Q[-1] < 100
    return sim


@sc.timer()
def test_births():
    sc.heading('Testing births and deaths, as compartments and as agents')
    def make():
        S, I, R = ss.states('S I R')
        return ss.Disease(
            infection = ss.Transition(S, I, rate=0.3 * I),
            recovery  = ss.Transition(I, R, dur=10),
            births    = ss.birth(S, rate=ss.peryear(0.5)),
            deaths    = ss.death(rate=ss.peryear(0.2)),
            init      = dict(I=0.01),
            name      = 'sir',
        )
    sims = sc.objdict()
    for solver in ['ode', 'abm']:
        pars = dict(n_agents=5_000) if solver == 'abm' else {} # The agents grow past the space first set aside for them
        sims[solver] = ss.Sim(make(), unit='day', stop=365, total_pop=10_000, solver=solver, **pars, **kw).run()
    n = sc.objdict({solver:sum(sim.results.sir[s][-1] for s in 'SIR') for solver,sim in sims.items()})
    assert np.isclose(n.ode, 10_000*np.exp(0.3), rtol=1e-3)
    assert np.isclose(n.abm, n.ode, rtol=0.03), f'Expected as many people as agents ({n.abm:n}) as as compartments ({n.ode:n})'
    assert np.isclose(sims.abm.results.sir.births.cum[-1], sims.ode.results.sir.births.cum[-1], rtol=0.05)

    # As agents, people who die in one disease leave the states of another on the same step
    S1, X1, S2, X2 = ss.states(4)
    one = ss.Disease(S1, X1, nothing=S1 >> X1 @ 0*S1.n, first='S1', name='one')
    two = ss.Disease(S2, X2, dying=S2 >> ss.null @ ss.perday(0.1), first='S2', name='two')
    res = ss.Sim([one, two], n_agents=1000, unit='day', stop=4, **kw).run().results
    assert np.allclose(res.one.S1, res.two.S2) and np.allclose(res.one.S1, res.alive)

    # Births and deaths of the population, in demographics=: the same people are born into and die from every disease, under every solver
    def flu():
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=ss.Transition(S, I, rate=ss.peryear(2) * I), recovery=ss.Transition(I, R, rate=ss.peryear(1)), init_prev=0.05, name='flu')
    sis = lambda: ss.SIS(beta=ss.peryear(0.5), dur_inf=ss.expon(ss.years(2)), name='sis')
    demog = [ss.birth(rate=ss.peryear(0.05)), ss.death(rate=ss.peryear(0.02))]
    for solver in ['ode', 'binomial', 'gillespie', 'abm']:
        sim = ss.Sim(flu(), sis(), demographics=demog, solver=solver, n_agents=2000, stop=20, **kw).run()
        n_flu = sum(sim.results.flu[s][-1] for s in 'SIR')
        n_sis = sum(sim.results.sis[s][-1] for s in 'SI')
        assert np.isclose(n_flu, n_sis), f'{solver}: {n_flu:n} people in flu but {n_sis:n} in sis' # Before, births only in one disease, or twice
        assert np.isclose(n_flu, 2000*np.exp(0.03*20), rtol=0.06) # 3,644
    assert ss.Sim(flu(), demographics=demog, **kw).init().solver == 'ode' # Not agents, as birth_rate= gave
    alone = ss.Sim(demographics=True, **kw).run() # With no declared disease, a declared module of everyone alive, chosen for as any declared model (it was ss.Births and ss.Deaths, as agents)
    assert alone.solver == 'ode' and alone.results.population.births.sum() > 0
    with pytest.raises(ValueError, match='people are born and die once'): # Births in two diseases would count people twice
        ss.Sim(ss.SIR(name='a', births=ss.birth(rate=0.02)), ss.SIS(name='b', births=ss.birth(rate=0.02)), **kw).init()

    # An inflow given for each level arrives at that level as given, whatever the split
    S, I = ss.states(2)
    arrivals = ss.Disease(arrive=ss.null >> S @ ss.by(sex=[100, 10]), infection=S >> I @ 0*I, name='arrivals')
    arrivals.stratify(sex=['f', 'm'], split=[1000, 500])
    res = ss.Sim(arrivals, total_pop=1500, stop=3, **kw).run().results.arrivals
    assert np.allclose(res.arrive.by('sex').values[1:], [100, 10]) # None at the start
    with pytest.raises(ValueError): # A birth rate for each level would be births from each level, which cannot be declared yet
        S, I = ss.states('S I')
        ss.Sim(ss.Disease(births=ss.birth(S, rate=ss.by(sex=[0.02, 0])), infection=ss.Transition(S, I, rate=0 * I), name='births').stratify(sex=['f', 'm']), **kw).init()

    # Migration: immigrants arrive at any age (and age group), and emigrants leave without dying
    for age in [False, True]:
        S, I = ss.states(2)
        mig = ss.Disease(S, I, nothing=S >> I @ 0*S.n, arrive=ss.immigration(S, rate=ss.peryear(500)), leave=ss.emigration(rate=ss.peryear(0.05)), name='mig')
        if age:
            mig.stratify(age=[0, 15, 50])
        sim = ss.Sim(mig, n_agents=10_000, total_pop=10_000, start=2000, stop=2010, **kw).run()
        res = sim.results
        assert res.new_deaths.sum() == 0 and res.new_emigrants.sum() == res.mig.leave.sum() > 0
        assert np.isclose(res.alive[-1], 10_000, rtol=0.05) # As many arrive as leave
        assert sim.people.age.mean() > 25 # Not newborns
        if age:
            assert res.mig.S.by('age').values[-1, -1] > 2000 # Arrivals over 50 are in the oldest group

    # An Euler step whose outflows would take more than everyone takes everyone, and no more
    S, I = ss.states(2)
    sim = ss.Sim(ss.Disease(fast=S >> I @ 5, init_n=dict(S=100), name='fast'), total_pop=100, start=2000, stop=2002, integrator='euler', substeps=1, **kw)
    assert np.allclose(sim.run().results.fast.I, [0, 100, 100]) # Everyone is infected in the first step
    return sims


@sc.timer()
def test_stratify():
    sc.heading('Testing stratification by age with a contact matrix')
    matrix = [[12, 4, 1], [3, 8, 2], [1, 3, 4]] # Contacts per day of each age group (rows) with each age group
    sims = sc.objdict()
    for solver in ['ode', 'binomial', 'abm']:
        sir = make_sir().stratify(age=[0, 15, 65], adjust={'recovery': [1.5, 1.0, 0.5]})
        sir.pars.beta.value = 0.03
        pars = dict(n_agents=20_000) if solver == 'abm' else {}
        sim = ss.Sim(sir, networks=ss.contacts(matrix, ages=[0, 15, 65]), total_pop=100_000, unit='day', stop=150, solver=solver, **pars, **kw).run()
        sims[solver] = sim
    with pytest.raises(TypeError): # The matrix's own age groups are ages=; there is no age=
        ss.contacts(matrix, age=[0, 15, 65])
    res = sims.ode.results.sir
    by_age = res.S.by('age')
    assert by_age.values.shape == (151, 3)
    assert np.allclose(by_age.values.sum(axis=1), res.S) # The strata sum to the total
    assert np.allclose(res.I.select(age=0), res.I.by('age').values[:, 0]) # By position
    assert np.allclose(res.I.select(age='65+'), res.I.by('age').values[:, 2]) # By label
    assert np.allclose(res.I.select(age=[0, 1]).by('age').values[:, 2], 0) # A selection can be split again
    S, I, R = ss.states(3)
    summed = ss.Disease(S, I, R, infection=S >> I @ 0.3*I, recovery=I >> R @ ss.perday(0.1), ever=I + R, init_prev=0.01, name='summed')
    summed.stratify(age=[0, 20])
    sres = ss.Sim(summed, unit='day', stop=40, **kw).run().results.summed
    assert np.allclose(sres.ever.by('age').values, (sres.I + sres.R).by('age').values) # A declared sum of states is stratified
    S, I, R = ss.states(3)
    ratio = ss.Disease(S, I, R, infection=S >> I @ 0.3*I, recovery=I >> R @ ss.perday(0.1), prev=I/(S+I+R), init_prev=0.01, name='ratio')
    ratio.stratify(age=[0, 20, 60])
    rres = ss.Sim(ratio, unit='day', stop=40, **kw).run().results.ratio
    young = ['0-19', '20-59']
    assert np.allclose(rres.prev.select(age=young), rres.I.select(age=young)/(rres.S + rres.I + rres.R).select(age=young)) # A ratio adds up its numerator and denominator over the groups selected, then divides
    assert np.allclose(ss.by(age=0.2).per_group(sims.ode), 0.2) # One value for every level
    assert (1/ss.by(sex=[0.5, 2])).values == [2, 0.5] and np.allclose((ss.by([[1, 2]], dims=['a', 'b'])*2).array, [[2, 4]]) # Arithmetic on values by group
    pop = (res.S + res.I + res.R).by('age').values[0]
    world = ss.data.world
    assert np.allclose(pop, 100_000*ss.data.age_split([0, 15, 65], world.ages, world.counts), rtol=0.01) and pop[2] < 0.15*pop.sum() # With no split given, the people's ages are the world's (not uniform to 100, which put 35% over 65)
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
    with pytest.raises(TypeError, match='both carry a time unit'): # beta and n_contacts both per unit time
        S, I = ss.states('S I')
        ss.Sim(ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.03) * I), init_prev=0.01), networks=ss.MixingPool(n_contacts=ss.perday(10)), **kw).init()
    S, I = ss.states('S I')
    approx = ss.Sim(ss.Disease(infection=ss.Transition(S, I, rate=0.1 * I), init_prev=0.1, name='d'), networks='random', solver='ode', stop=10, **kw).init()
    with pytest.raises(AttributeError, match='all_networks'): # A network that a compartmental run approximates is not one of its networks, which it says
        approx.networks.randomnet

    # People given to a compartmental run give its age split and size, as they would as agents
    ppl = lambda: ss.People(10_000, age_data=np.array([np.arange(80), np.exp(-0.03*np.arange(80))]).T)
    def young(solver):
        sir = make_sir().stratify(age=[0, 15, 65])
        n_S = ss.Sim(sir, people=ppl(), unit='day', stop=10, solver=solver, **kw).run().results.sir.S
        return n_S.by('age').values[0, 0]/n_S[0]
    assert np.isclose(young('binomial'), young('abm'), atol=0.02) # About 40% under 15

    # Stratifying the sim gives every disease the same groups
    def make(name, beta, dur):
        S, I = ss.states('S I')
        return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, S, dur=dur), init_prev=0.01, name=name)
    pars = dict(unit='day', stop=100, **kw)
    both = ss.Sim([make('flu', 0.4, 5), make('rsv', 0.3, 8)], **pars).stratify(age=[0, 5, 18, 65]).run()
    same = ss.Sim([make('flu', 0.4, 5), make('rsv', 0.3, 8)], stratify=dict(age=[0, 5, 18, 65]), **pars).run()
    assert ss.debugtools.check_sims_match(both, same)
    for key in ['flu', 'rsv']:
        assert both.results[key].I.by('age').values.shape == (101, 4)
    assert both.new().run().results.flu.I.by('age').values.shape == (101, 4) # A new sim is stratified too

    # Errors
    again = make_sir().stratify(age=[0, 15]).stratify(age=[0, 15]) # The same line run twice does nothing more
    assert len(again.graph.strats) == 1
    assert ss.Sim(make_sir(), **kw).stratify(age=[0, 15]).stratify(age=[0, 15]).init().groups.n_groups == 2 # Likewise for the sim
    with pytest.raises(ValueError, match='is already stratified by "age", in another way'): # In another way, it is an error
        make_sir().stratify(age=[0, 15]).stratify(age=[0, 20])
    with pytest.raises(KeyError, match='adjust= names a transition "recovry"'):
        make_sir().stratify(age=[0, 15], adjust={'recovry': [1, 2]})
    with pytest.raises(ValueError, match='split= has 3 entries'):
        make_sir().stratify(risk=['low', 'high'], split=[0.5, 0.3, 0.2])
    with pytest.raises(ValueError, match='has 2 values, but "age" has 3 levels'): # Caught when the sim is built, not when it runs
        S, I = ss.states('S I')
        ss.Sim(ss.Disease(infection=ss.Transition(S, I, rate=ss.by(age=[0.3, 0.2]) * I), init_prev=0.01), stratify=dict(age=[0, 15, 65]), **kw).init()
    with pytest.raises(ValueError, match='A property of people has the same groups for every module'):
        a = make_sir().stratify(age=[0, 15])
        I, R = ss.states('I R')
        b = ss.Disease(recovery=ss.Transition(I, R, dur=3), name='other').stratify(age=[0, 20])
        ss.Sim([a, b], **kw).init()

    # One parser for the sim and the disease: an option is never a dimension, and the sim's adjust= is checked too
    assert [strat.name for strat in ss.Sim(make_sir(), stratify=dict(age=[0, 15], states=['S', 'I']), **kw).init().groups.strats] == ['age'] # Not a dimension called states
    with pytest.raises(KeyError, match='that no single module has'):
        ss.Sim(make_sir(), stratify=dict(age=[0, 15], adjust={'recovry': [1, 2]}), **kw).init()
    with pytest.raises(ValueError):
        make_sir().stratify(split=[0.5, 0.5]) # No dimension

    # A dimension called immunity, with its split used even though the sim has the same groups
    imm = ss.Sim(make_sir().stratify(immunity=['none', 'high'], split=[0.7, 0.3]), stratify=dict(immunity=['none', 'high']), unit='day', stop=10, **kw).run()
    n = sum(imm.results.sir[k].by('immunity').values[-1] for k in 'SIR')
    assert np.allclose(n/n.sum(), [0.7, 0.3]) # Not no dimension, nor the sim's equal split
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
    assert np.isclose(res.prevalence.max(), res.I.max()/1e6)

    # Fit beta to the weekly cases, which are compared with the total over each week
    weekly = res.cases.resample('week')
    df = sc.dataframe(date=weekly.timevec, cases=weekly.values)
    sim = ss.Sim(make(0.15), data=df, **pars)
    fit = sim.fit()
    fit.summarize()
    assert np.isclose(fit.pars.beta, 0.3, rtol=0.01), f'Expected to recover beta=0.3, not {fit.pars.beta}'
    assert fit.mismatch < 0.01 < fit.mismatch_before
    assert fit.converged and not sim.fit(method='nelder-mead', maxiter=3, scan=0, verbose=0).converged
    assert np.isclose(sim.new().init().diseases.sir.pars.beta.value, 0.15) # The original is not changed
    zeros = ss.fit.mismatch(truth, fit.sim.data.assign(deaths=0.0)) # An all-zero series is scaled by the model, not counted raw
    assert np.isclose(zeros, 0.5) # Cases exact (0), deaths 1; before, the raw count of deaths (about 420 a week) swamped the cases
    refit = ss.Sim(make(0.15), pars=fit.pars, **pars).run() # The best-fit values can be passed to another sim
    assert np.allclose(refit.results.sir.I, fit.sim.results.sir.I)
    named = ss.Sim(make(0.3), pars=dict(dur=ss.days(5), rand_seed=2), **pars).run() # A parameter named like a sim setting is the model's; a sim setting is the sim's
    assert named.results.sir.I.max() < 0.5*res.I.max() and named.pars.rand_seed == 2 # Before, dur= was taken as the sim's duration, and changed nothing
    assert ss.Sim(ss.SIS(), ss.RandomNet(), ss.SexualNet(), pars=dict(dur=10), n_agents=100, verbose=0).init().pars.dur == 10 # But a setting of networks written as code is not fitted, so dur= is the sim's

    short = ss.Sim(make(0.3), data=df, **(pars | dict(stop='2020-03-01'))).init()
    assert any('outside the simulated period' in note for note in short.notes) # Data that the comparison leaves out is noted

    # Mistakes in the data and the parameters are caught by name
    typo = ss.Sim(make(0.3), data=df.rename(columns=dict(cases='casess')), **pars).init()
    assert any('casess' in note for note in typo.notes) # Data on a sim can be just for plotting, so a column that matches no result is noted, and an error only when fitting
    with pytest.raises(KeyError, match='the data column "casess" matches no result'):
        typo.fit()
    renamed = ss.simdata(df.rename(columns=dict(cases='Reported')), aliases=dict(cases='Reported'))
    assert 'cases' in ss.Sim(make(0.3), data=renamed, **pars).init().data.columns
    assert 'cases' in ss.Sim(make(0.3), **pars).new(data=renamed).init().data.columns # Or to a new sim
    with pytest.raises(KeyError, match='pars= has a value for "betta", which matches no parameter'):
        ss.Sim(make(0.3), pars=dict(betta=0.2), **pars).init()
    assert ss.expressions.did_you_mean('betta', ['sir.beta', 'sir.dur']) == '; did you mean sir.beta?' # The closest only, matched without the module's name
    with pytest.raises(ValueError, match='outside its bounds'): # A value outside its bounds
        ss.Sim(make(0.3), pars=dict(beta=5), **pars).init()
    with pytest.raises(ValueError, match='has no value'): # A parameter with no value
        S, I, R = ss.states('S I R')
        ss.Sim(ss.Disease(infection=ss.Transition(S, I, rate=ss.par() * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01), **pars).init()
    with pytest.raises(ValueError, match=r'fit\(\) needs data'):
        ss.Sim(make(0.3), **pars).fit()
    with pytest.raises(ValueError, match='gof'): # Rather than the default in its place
        sim.fit(gof='nea')
    with pytest.raises(KeyError, match='is not a transition or outcome of "ir"'):
        I, R = ss.states('I R')
        ss.Sim(ss.Disease(recovery=ss.Transition(I, R, dur=3), cases=ss.outcome('infection'), first='I'), **kw).init()

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
        sir = ss.Disease(infection, I >> R @ 0.1, ss.outcome(0.5 * infection, name='cases', delay=ss.days(3)), ss.outcome(infection, name='hosp', duration=ss.days(5)), init_n=dict(I=ss.by(patch=[5, 0])), name='sir')
        sir.stratify(patch=['a', 'b'], mobility=ss.geo.Mobility(matrix=[[0, 0.01], [0.01, 0]], kind='migrate'))
        return ss.Sim(sir, total_pop=1000, unit='day', stop=80, **kw)
    kernel, stepped = by_patch().run().results.sir, by_patch().run(profile=True).results.sir # Recording the time of each step needs the loop
    agents = by_patch().new(n_agents=1000).run().results.sir
    for res in [kernel, stepped, agents]:
        assert np.allclose(res.cases.select(patch='a') + res.cases.select(patch='b'), res.cases)
        assert np.allclose(res.n_hosp.by('patch').values.sum(axis=1), res.n_hosp)
    assert np.allclose(kernel.cases.select(patch='b'), stepped.cases.select(patch='b'))
    assert np.isclose(agents.cases.select(patch='b').sum(), kernel.cases.select(patch='b').sum(), rtol=0.3)

    # A weight for each group on the flow an outcome counts, in a whole run in the kernels
    S, E, I, R = ss.states('S E I R')
    progression = ss.Transition(E, I, rate=0.5)
    weighted = ss.Disease(infection=ss.Transition(S, E, rate=2 * I), onset=progression, recovery=ss.Transition(I, R, rate=0.3),
                          notified=ss.outcome(ss.by(age=[0, 1])*progression), init_n=dict(I=10), name='weighted').stratify(age=['young', 'old'])
    wres = ss.Sim(weighted, stop=50, **kw).run().results.weighted
    assert np.isclose(wres.notified.sum(), wres.onset.select(age='old').sum())

    # And as agents, a weight by age group or by a function of each person (it crashed); the ODE takes the weight by group
    def yll(weight, solver='abm'):
        S, I, R = ss.states('S I R')
        death = I >> ss.null @ ss.peryear(2)
        d = ss.Disease(S, I, R, infection=S >> I @ ss.perday(0.2)*I, recovery=I >> R @ ss.days(20), death=death, init=dict(I=0.05), yll=ss.outcome(weight*death), name='d')
        d.stratify(age=[0, 15, 50, 70])
        return ss.Sim(d, n_agents=2000, total_pop=2000, solver=solver, dt=ss.days(7), start=2000, stop=2001, **kw).run().results.d
    res = yll(ss.by(age=[60, 50, 30, 10]))
    deaths, lost = res.death.by('age').values.sum(0), res.yll.by('age').values.sum(0)
    assert deaths.sum() > 100 and np.allclose(lost, deaths*[60, 50, 30, 10])
    res = yll(lambda sim, uids: np.maximum(90 - sim.people.age[uids], 0))
    assert 0 < res.yll.sum() < 90*res.death.sum()
    ode = yll(ss.by(age=[60, 50, 30, 10]), solver='ode')
    assert np.isclose(ode.yll.sum()/ode.death.sum(), lost.sum()/deaths.sum(), rtol=0.1)

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
    shared = ss.Disease(infection=infection, recovery=I >> R @ ss.days(10), cases=0.5*infection, init_n=dict(I=10), name='shared')
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
        sir = ss.Disease(infection, *stages, I >> R @ ss.days(7), cases, prevalence=I/N, init_n=dict(I=10), name='sir')
        return ss.Sim(sir, total_pop=10_000, unit='day', stop=ss.weeks(20), label='SEIR' if exposed else 'SIR', **kw)

    # Data drawn from a negative binomial around the weekly cases of an SEIR
    truth = make(exposed=True).run().results.sir
    weekly = truth.cases.resample('week')
    rng = np.random.default_rng(1)
    df = sc.dataframe(t=weekly.timevec, cases=rng.negative_binomial(10, 10/(10 + weekly.values)))

    # The declared likelihood is used, and its dispersion is fitted with beta
    fit = make(beta=0.2, k=3, exposed=True).fit(data=df, method='l-bfgs-b') # The maximum likelihood (NUTS is tested in test_gradients.py)
    fit.summarize()
    assert fit.measure == 'negative log-likelihood'
    assert np.isclose(fit.pars.beta.value, 0.4, rtol=0.05)
    assert 2 < fit.pars.k < 100
    assert np.isclose(ss.fit.loglik(fit.sim), -fit.mismatch)
    pulled = make(beta=0.2, k=3, exposed=True, prior=ss.normal(0.3, 0.002)).fit(data=df, method='l-bfgs-b') # With a prior, the most probable value
    assert pulled.measure == 'negative log-posterior' and pulled.pars.beta.value < 0.35

    # Counts that are not whole numbers (e.g. weekly averages) have a likelihood, and data that no run can give stops the fit
    averaged = make(beta=0.2).fit(data=df.assign(cases=df.cases/7), method='l-bfgs-b')
    assert np.isfinite(averaged.mismatch) and averaged.pars.beta.value > 0.2
    with pytest.raises(ValueError, match='cases'):
        make().fit(data=df.assign(cases=-df.cases/7), method='l-bfgs-b')

    # A likelihood can be given at calibration time, for a result that is not an outcome
    prev = sc.dataframe(t=truth.timevec[::7], prevalence=truth.prevalence.values[::7])
    fit2 = make(beta=0.2, exposed=True).fit(data=prev, likelihood=dict(prevalence=ss.normal(scale=0.01)), method='l-bfgs-b')
    assert fit2.measure == 'negative log-likelihood'
    assert np.isclose(fit2.pars.beta.value, 0.4, rtol=0.01)

    # The model that made the data predicts the held-out weeks better
    comp = ss.fit.compare_models([make(), make(exposed=True)], data=df, verbose=0) # verbose= is passed to each fit (it was given twice)
    assert comp.table.index[0] == 'SEIR'
    assert list(comp.fits.keys()) == ['SIR', 'SEIR']
    mixed = ss.fit.compare_models(dict(lik=make(), nolik=make(lik=False)), data=df).table # One measure for both, although only one has a likelihood
    assert np.isclose(mixed.fit_error['lik'], mixed.fit_error['nolik'])

    # Data with its dates as text can be plotted with the sim
    dated = ss.Sim(make_sir(), start='2020-01-01', stop='2020-02-01', data=pd.DataFrame(dict(date=['2020-01-08', '2020-01-15'], infection=[5, 9])), **kw).run()
    dated.plot()

    # Parameters can be fitted inside an initial count and inside a seasonal value
    def seasonal(I0, smin):
        S, E, I, R = ss.states(4)
        season = ss.seasonal(mean=(1 + smin)/2, amplitude=(1 - smin)/(1 + smin), peak='January')
        return ss.Sim(ss.Disease(infection=S >> E @ 0.4*season*I, onset=E >> I @ ss.days(3), recovery=I >> R @ ss.days(7), init_n=dict(E=I0/2, I=I0), name='x'),
                      start='2020-01-01', stop='2020-07-01', unit='day', **kw)
    truth = seasonal(ss.par(80.0), ss.par(0.3)).run()
    data = pd.DataFrame(dict(date=truth.t.datevec[::7], I=truth.results.x.I.values[::7])).set_index('date')
    fit3 = seasonal(ss.par(20.0, bounds=[1, 200]), ss.par(0.7, bounds=[0.1, 1])).fit(data=data)
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
    n_R = msim.results.sir.R
    assert [sim.label for sim in msim.sims] == ['baseline', 'masks']
    assert n_R.select(scenario='masks').values[0, -1] < 0.8*n_R.select(scenario='baseline').values[0, -1]
    assert np.allclose(n_R.select(scenario='masks').values[0, :20], n_R.select(scenario='baseline').values[0, :20]) # The same until it starts
    assert 'masks' in msim.sims[1].interventions # Named from its variable
    assert np.isclose(sir.pars.beta.value, 0.3) # The base value is never overwritten
    timed = lambda when: ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.5, when=when), **pars)
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
        base, faster = [ss.Sim(sir, stop=40, solver=solver, interventions=mods, **agents, **kw).run().results.sir.I[25] for mods in [None, ss.modify(sir.recovery, multiply=2, start=20)]]
        halves[solver] = faster/base
    assert abs(halves['abm'] - halves['ode']) < 0.1, f'Expected the same effect as agents and as compartments, not {halves}'

    # So do a modifier on its duration and a duration that changes in time, as agents: a cohort, 498 left in the ODE on day 25 or 15 (not 825 and 4,755)
    def cohort(solver, dur, mods=None):
        I, R = ss.states('I R')
        dis = ss.Disease(recovery=I >> R @ dur, first='I', name='sir')
        size = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
        return ss.Sim(dis, interventions=mods(dis) if mods else None, unit='day', dt=0.1, stop=40, solver=solver, rand_seed=1, verbose=0, **size).run().results.sir.I.values
    shorter = lambda dis: ss.modify(dis.pars.dur, multiply=0.5, start=20)
    varying = ss.timeseries({0: ss.days(20), 10: ss.days(2)})
    for dur, mods, i in [(ss.days(10), shorter, 250), (varying, None, 150)]:
        ode, abm = cohort('ode', dur, mods)[i], cohort('abm', dur, mods)[i]
        assert np.isclose(abm, ode, rtol=0.1), f'agents {abm:n}, ODE {ode:n}'

    # A where= on an exit with a dwell time that becomes true while people wait: they leave once it holds, a year later
    A, B = ss.states('A B')
    aging = ss.Disease(A, B, ab=ss.Transition(A, B, dur=ss.constant(ss.years(1)), where=lambda sim: sim.people.age.raw >= 30), first='A', name='aging')
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
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=ss.seasonal(mean=0.3, amplitude=0.2) * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
        mods = [ss.modify(sir.pars.beta, multiply=0.5, start=30, stop=60), ss.change(sir.recovery, at=[40], by=[2.0]), ss.modify(sir.rel_sus, multiply=0.8, start=50)]
        return ss.Sim(sir, interventions=mods, unit='day', stop=120, **kw).init() # A yearly cycle needs a unit
    fast, slow = timed(), timed()
    assert fast.kernel_only()
    assert np.allclose(fast.run().results.sir.I, slow.run(profile=True).results.sir.I)

    # A modifier can also be given as a path, and combines with others by multiplication
    both = [ss.modify('sir.pars.beta', multiply=0.5, start=20, name='a'), ss.modify('sir.pars.beta', multiply=0.5, start=20, name='b')]
    twice = ss.Sim(sir, interventions=both, **pars).run()
    assert twice.results.sir.R[-1] < n_R.select(scenario='masks').values[0, -1]
    with pytest.raises(KeyError, match='sir.pars.betta, was not found'):
        ss.Sim(sir, interventions=ss.modify('sir.pars.betta', multiply=0.5), **pars).init()

    # A module's array as a modifier's amount (it crashed, saying a state was already added), and a modifier named like a property of people is refused
    class Coded(ss.Disease):
        def __init__(self, **kwargs):
            super().__init__(name='coded', **kwargs)
            self.define_states(ss.FloatArr('factor', default=2.0))
        def step(self): pass
    S, I = ss.states('S I')
    sis = ss.Disease(name='sis', infection=ss.Transition(S, I, rate=ss.peryear(0.5) * I), recovery=ss.Transition(I, S, dur=ss.years(1)), init=dict(I=0.1))
    c = Coded()
    agents = dict(n_agents=1000, start=2000, stop=2003, dt=ss.years(0.25), **kw)
    ss.Sim(diseases=[c, sis], networks='random', connectors=ss.modify(sis.rel_sus, multiply=c.factor), **agents).run()
    S, I = ss.states('S I')
    d = ss.Disease(infection=ss.Transition(S, I, rate=0.1 * I), init_prev=0.1, name='d')
    with pytest.raises(AttributeError, match='another name'):
        ss.Sim(d, people=ss.People(1000, extra_states=[ss.FloatArr('nab_imm')]), connectors=ss.modify(d.rel_sus, multiply=0.5, name='nab_imm'), start=2000, stop=2001, **kw).run()

    # adjust= on a branch chosen with p= scales its probability, as ss.modify does, under every solver (it divided the dwell; the ODE read a share by a dimension split on entry at its first level)
    def branch_share(solver, adjust=None, modify=False):
        S, D, R = ss.states('S D R')
        d = ss.Disease(S, D, R, onset=S >> D @ ss.peryear(2), death=ss.Transition(D, ss.null, dur=ss.days(10), p=0.1), recovery=ss.Transition(D, R, dur=ss.days(10)), back=R >> S @ ss.days(5), name='d')
        d.stratify(type=['watery', 'bloody'], states=[D], adjust=adjust)
        con = ss.modify(d.death, multiply=5, where=d.type == 'bloody') if modify else None
        res = ss.Sim(d, connectors=con, n_agents=20_000, total_pop=20_000, solver=solver, dt=ss.days(1), start=2000, stop=2001, **kw).run().results.d
        return res.death.by('type').values.sum(0)/res.onset.by('type').values.sum(0)
    for solver in ['ode', 'binomial', 'abm']:
        for share in [branch_share(solver, adjust={'death': [1, 5]}), branch_share(solver, modify=True)]:
            assert np.allclose(share, [0.1, 0.5], atol=0.02), f'{solver}: {share}'

    # On one transition with several destinations and a dwell, a modifier changes the dwell, and a note names the destinations that change a branch
    G, A, B = ss.states('G A B')
    d = ss.Disease(G, A, B, go=ss.Transition(G, [A, B], dur=ss.weeks(4), p=[0.95, 0.05]), first='G', name='d')
    sim = ss.Sim(d, connectors=ss.modify('d.go', multiply=4), total_pop=10_000, solver='ode', start=2000, stop=2001, **kw).run()
    assert any("'d.go.B'" in note for note in sim.notes)
    sim = ss.Sim(d, connectors=ss.modify('d.go.B', multiply=4), total_pop=10_000, solver='ode', start=2000, stop=2001, **kw).run()
    assert np.isclose(sim.results.d.B[-1], 2000, rtol=1e-3)

    # One sim is one run; many runs are a MultiSim
    sim = ss.Sim(sir, solver='binomial', total_pop=1000, stop=60, **kw)
    runs = ss.parallel(sim, n_runs=8)
    n_I = runs.results.sir.I
    assert n_I.values.shape == (8, 61) and len(runs) == 8
    assert len(np.unique(n_I.values[:, 30])) > 1 # Each run has its own seed
    assert runs.summarize().loc['sir_I', 'std'] > 0 # Summarized over the eight runs (before, over one sim: a standard deviation of 0)
    assert np.array_equal(n_I.select(run=3).values, n_I.values[3]) and np.allclose((n_I + 1).values, n_I.values + 1) # A result like any other
    low, mid, high = n_I.quantiles([0.025, 0.5, 0.975])
    assert np.all(low <= mid) and np.all(mid <= high)
    assert sim.run(n_runs=3).results.sir.I.values.shape == (3, 61) # The same thing, from the sim
    assert len(runs.sims) == 8 and np.allclose([s.results.sir.I[30] for s in runs.sims], n_I.values[:, 30]) # The sim of each run, made when asked for

    # A parameter swept by its full name is selected by its own, and a sweep's scenarios are named by it
    sweep = ss.parallel(ss.Sim(make_sir(), stop=30, **kw), pars={'sir.beta': [0.2, 0.4]})
    assert sweep.results.sir.I.select(beta=0.4).values.max() > sweep.results.sir.I.select(beta=0.2).values.max()
    S, I = ss.states('S I')
    d = ss.Disease(infection=ss.Transition(S, I, rate=ss.par(0.1, name='beta') * I), init_prev=0.1, name='d')
    sweep = ss.parallel(ss.Sim(d, n_agents=1000, start=2000, stop=2003, dt=ss.years(0.25), **kw), pars=dict(beta=[0.1, 0.2]), n_runs=1)
    assert list(sweep.results.d.I.scenarios()) == ['beta=0.1', 'beta=0.2']
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
        S, I, H, R = ss.states('S I H R')
        d = ss.Disease(
            infection = ss.Transition(S, I, rate=0.3 * I),
            course    = ss.Transition(I, [H, R], p=severity, dur=7),
            discharge = ss.Transition(H, R, rate=0.1),
            init_prev = 0.01,
            name = 'd',
            **kwargs,
        )
        return d

    # Code is not run to choose the solver: with no age groups the function cannot run as compartments, which it says, so the model is run as agents by asking
    with pytest.raises(ss.CapabilityError, match='people.age'):
        ss.Sim(make(), stop=100, unit='day', **kw).run()
    sim = ss.Sim(make(seeding=seed_cluster), stop=100, unit='day', solver='abm', **kw).run()
    assert sim.results.d.I[-1] == sim.diseases.d.I.count() # The counts kept through the run, which the function's changes to the states are counted in, are those of the agents
    by_age = ss.Sim(make(), stop=100, unit='day', stratify=dict(age=[0, 50]), **kw).run()
    assert by_age.solver == 'ode'
    res = by_age.results.d
    assert res.H.select(age='50+').max() > res.H.select(age='0-49').max() # Older people are more likely to be hospitalized
    with pytest.raises(ss.CapabilityError, match='falls inside a group'): # 50 is inside a group, which it says before the run (it ran as an ODE, giving each group the age of its middle)
        ss.Sim(make(), stop=100, unit='day', stratify=dict(age=[0, 15, 65]), **kw).init()
    with pytest.raises(ss.CapabilityError):
        ss.Sim(make(), stop=100, unit='day', solver='ode', **kw).run()
    def typo(sim):
        return sim.results.d['X'][sim.ti] # A mistake, which agents would not fix
    with pytest.raises(KeyError, match='X'):
        S, I, H, R = ss.states('S I H R')
        bad = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), course=ss.Transition(I, [H, R], p=typo, dur=7), discharge=ss.Transition(H, R, rate=0.1), name='d')
        ss.Sim(bad, stop=100, unit='day', **kw).run()

    # Code of the user's own (a function given as an intervention, a step() of a declared disease) runs beside the compartments, and compartmental=False says it needs agents
    def knock(sim):
        if sim.ti == 20:
            sim.diseases.sir.pars.beta = 0.1
    def sir():
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
    knocked = ss.Sim(sir(), interventions=knock, stop=60, unit='day', **kw).run()
    assert knocked.solver == 'ode' # Not 10,000 agents
    assert knocked.results.sir.infection.cum[-1] < 0.7*ss.Sim(sir(), stop=60, unit='day', **kw).run().results.sir.infection.cum[-1] # It still acts
    def stop_in_place(sim):
        if sim.ti == 20:
            sim.diseases.sir.pars.beta.raw.value = 0.0 # The number inside the typed rate, changed in place
    S, I, R = ss.states('S I R')
    typed = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3) * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
    assert ss.Sim(typed, interventions=stop_in_place, stop=60, unit='day', **kw).run().results.sir.infection[22] == 0 # Seen (it was not)
    def seed(sim): # Per-step behavior that works on groups
        if sim.ti == 3:
            sir = sim.diseases.sir
            ss.move(sir.S, sir.I, n=5)
    def seed_agents(sim): # Per-step behavior that needs agents
        sim.diseases.sir.S.uids
    def seeded(fn):
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=ss.Transition(S, I, rate=0 * I), recovery=ss.Transition(I, R, dur=1000), init=dict(I=0), seeding=fn, name='sir')
    sim = ss.Sim(seeded(seed), stop=6, unit='day', **kw).run()
    assert sim.solver == 'ode' and sim.results.sir.I[-1] > 4.9 # Called each step, not skipped by a whole run (was 0)
    assert ss.Sim(seeded(seed), stop=6, unit='day', solver='abm', **kw).run().results.sir.I[-1] == 5 # The same code, as agents
    def cure(sim): # People are moved, never made or lost: the number of people stays the same in every disease
        sir = sim.diseases.sir
        if sim.ti == 10:
            assert ss.move(sir.I, sir.R, p=0.5) > 0 and ss.move([sir.S, sir.R], sir.I, n=7) == 7 # Exactly n, as whole people too
            ss.move(ss.null, sir.W, n=100) # A count can come from outside the model
            with pytest.raises(ss.CapabilityError, match='ss.move'): # Setting the number of people in a state would make or lose people
                sir.I.n = 0
            with pytest.raises(ValueError, match='so the number of people stays the same'):
                ss.move(sir.I, ss.null)
    for solver in ['ode', 'binomial', 'abm']:
        S, I, R, W = ss.State(name='S'), ss.State(name='I'), ss.State(name='R'), ss.State(name='W', people=False)
        dying = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(7), I >> ss.null @ ss.perday(0.01), W >> ss.null @ ss.perday(0.5), init=dict(I=0.01), name='sir')
        S, I = ss.State(name='S'), ss.State(name='I')
        sis = ss.Disease(S >> I @ 0.2*I, I >> S @ ss.days(10), init=dict(I=0.01), name='sis')
        sim = ss.Sim(diseases=[dying, sis], interventions=cure, solver=solver, total_pop=2000, n_agents=2000, unit='day', stop=20, **kw).run()
        res = sim.results
        assert np.allclose(res.sir.S + res.sir.I + res.sir.R, res.sis.S + res.sis.I) and res.sir.W.values.max() > 50
    with pytest.raises(ss.CapabilityError, match='uids'): # It asks for agents, which it says
        ss.Sim(seeded(seed_agents), stop=6, unit='day', **kw).run()
    class Mine(ss.Disease):
        def step(self):
            return super().step()
    class Agentish(Mine):
        compartmental = False
    def mine(cls):
        S, I, R = ss.states('S I R')
        return cls(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='mine')
    assert ss.Sim(mine(Mine), unit='day', **kw).init().solver == 'ode'
    assert ss.Sim(mine(Agentish), unit='day', **kw).init().solver == 'abm'
    births = ss.Sim(sir(), demographics=ss.Births(), unit='day', **kw).init()
    assert births.solver == 'abm' and 'ss.birth()' in births.solver_reason
    assert births.solver_reason in births.describe(output=True) # describe() says what auto did

    # A function that takes uids only works on agents
    S, I, R = ss.states('S I R')
    d2 = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, rate=clearance), init_prev=0.01, name='d2')
    sim2 = ss.Sim(d2, stop=100, unit='day', **kw).run()
    assert sim2.solver == 'abm' and 'clearance' in sim2.solver_reason
    assert 'takes uids' in sim2.describe(output=True)

    # A rate given as a function of the sim with one value per person alive, once people have died (it raised IndexError)
    A, B = ss.states('A B')
    d = ss.Disease(A, B, go=A >> B @ ss.peryear(0.5)*(lambda sim: 1.0*np.asarray(sim.people.female)), first='A', name='d')
    sim6 = ss.Sim(d, demographics=ss.Deaths(death_rate=ss.peryear(200)), n_agents=5000, dt=0.25, start=2000, stop=2001, **kw).run()
    inB = sim6.diseases.d.B.uids
    assert len(inB) > 100 and sim6.people.female[inB].all()

    # A state's .n and .count(), which functions read, under both solvers
    for solver in ['abm', 'ode']:
        S, I = ss.states('S I')
        counted = ss.Sim(ss.Disease(infection=ss.Transition(S, I, rate=0.1 * I), init_prev=0.1, name='d'), solver=solver, n_agents=1000, total_pop=10_000, **kw).init()
        assert counted.diseases.d.I.n == 1000 and counted.diseases.d.I.count() in [100, 1000]

    # Arguments with defaults do not count: this is a function of the sim
    def recover(sim, factor=2):
        return 0.1*factor
    S, I, R = ss.states('S I R')
    d3 = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, rate=recover), init_prev=0.01, name='d3')
    sim4 = ss.Sim(d3, stop=100, unit='day', **kw).run()
    assert sim4.solver == 'ode'
    eligible = lambda sim, years=5: sim.diseases.d3.I # Likewise for a condition
    sim5 = ss.Sim(d3.copy(), interventions=ss.treat(ss.Treatment(cure=1.0), where=eligible, coverage=0.5, name='tx'), stop=20, unit='day', n_agents=1000, **kw).run()
    assert sim5.results.tx.delivered.sum() > 0

    # A sexual network, with a transmission probability per act for each direction
    S, I = ss.states('S I')
    hiv = ss.Disease(infection=ss.Transition(S, I, rate=dict(sexual=ss.peract(m2f=0.008, f2m=0.004)) * I), init_prev=0.05, name='hiv')
    sim3 = ss.Sim(hiv, networks=ss.SexualNet(), start=2000, stop=2020, **kw).run()
    assert sim3.solver == 'abm' and 'SexualNet' in sim3.solver_reason
    assert sim3.unit == 'year'
    n_I = sim3.results.hiv.I
    assert n_I[-1] > 1.5*n_I[0]
    with pytest.raises(ValueError, match='no route is called needles'):
        S, I = ss.states('S I')
        other = ss.Disease(infection=ss.Transition(S, I, rate=dict(needles=0.1) * I), init_prev=0.05, name='hiv')
        ss.Sim(other, networks=ss.SexualNet(), start=2000, stop=2005, **kw).run()

    # An Erlang dwell time is exact in every solver
    S, I, R = ss.states('S I R')
    erl = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=ss.erlang(k=3, mean=ss.days(10))), init_prev=0.01, name='erl')
    S, I, R = ss.states('S I R')
    stg = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=ss.days(10), stages=3), init_prev=0.01, name='stg')
    r1 = ss.Sim(erl, unit='day', stop=100, **kw).run()
    r2 = ss.Sim(stg, unit='day', stop=100, **kw).run()
    assert not r1.notes # No approximation
    assert np.allclose(r1.results.erl.I, r2.results.stg.I)

    # A branch with p= and where= fires once people are in where= (before, never), and a function p= leaves the rest to the timed exit with none (before, 0.45 instead of 0.9)
    due = lambda sim: sim.people.uid.raw*0 + (sim.ti >= 5) # Everyone, from step 5
    I, R = ss.states('I R')
    gated = ss.Disease(recovery=ss.Transition(I, R, p=1.0, rate=ss.perday(1e6), where=due), first='I', name='gated')
    assert ss.Sim(gated, n_agents=1000, stop=10, unit='day', verbose=0).run().results.gated.R[-1] == 1000
    for solver in ['ode', 'abm']:
        E, A, Y = ss.states(3)
        d = ss.Disease(E=E, A=A, Y=Y, symp=ss.Transition(E, Y, p=lambda sim: 0.9, dur=ss.weeks(1)), asymp=ss.Transition(E, A, dur=ss.weeks(1)), first='E', name='d')
        res = ss.Sim(d, n_agents=10_000, total_pop=10_000, start=2000, stop=2001, dt=1/52, solver=solver, verbose=0).run().results.d
        assert np.isclose(res.symp.sum()/(res.symp.sum() + res.asymp.sum()), 0.9, atol=0.01)
    return sim


@sc.timer()
def test_time_values():
    sc.heading('Testing values that vary in time')
    def make(beta):
        S, I = ss.states('S I')
        return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, S, dur=10), init_prev=0.01, name='sis')
    dates = dict(start='2020-01-01', stop='2020-06-01', **kw)

    # A time series is a step function by default, so matches the same changes made with ss.change()
    series = ss.Sim(make(ss.timeseries({'2020-01-01': 0.3, '2020-02-01': 0.1, '2020-04-01': 0.3})), **dates).run()
    sis = make(0.3)
    changed = ss.Sim(sis, interventions=ss.change(sis.pars.beta, at=['2020-02-01', '2020-04-01'], to=[0.1, 0.3]), **dates).run()
    assert series.solver == 'ode' # A value that varies in time does not make a model agent-based
    assert np.allclose(series.results.sis.I, changed.results.sis.I)
    assert series.diseases.sis.pars.beta.raw.mode == 'constant'

    # With no dates, the times are in the sim's unit; and it can be the amount of a modifier
    sis = make(0.3)
    ramp = ss.modify(sis.pars.beta, multiply=ss.timeseries({0: 1.0, 50: 0.2}, mode='linear'))
    res = ss.Sim(sis, interventions=ramp, stop=100, **kw).run().results.sis
    assert res.I[:40].max() > 2*res.I[-1] # Transmission falls below what is needed to sustain it
    with pytest.raises(ValueError):
        ss.Sim(make(ss.timeseries({50: 1.0, 0: 0.2})), stop=10, **kw).run()

    # A time series inside ss.by: the same as a function that gives the value for each group, run in the kernels or not
    def by_age(beta):
        sis = make(beta)
        sis.stratify(age=[0, 20])
        return ss.Sim(sis, stop=100, **kw).run().results.sis.I
    series = by_age(ss.by(age=[ss.timeseries({0: 0.3, 50: 0.1}), 0.2]))
    func = by_age(lambda sim: ss.by(age=[0.3 if sim.t.now() < 50 else 0.1, 0.2]).per_group(sim))
    assert np.allclose(series, func)

    # A rate times a trend in time or a function of the sim, in either order, also as the deaths of everyone (DuplicateDeclaration and IncompatibleTypes before)
    trend = ss.timeseries({2000: 1.0, 2005: 0.5})
    deaths = [ss.Sim(make_sir(), demographics=ss.death(rate=rate), start=2000, stop=2010, total_pop=1000, **kw).run().results.sir.deaths.values.sum()
              for rate in [ss.peryear(0.02)*trend, trend*ss.peryear(0.02), ss.peryear(0.02)*(lambda sim: 1.0)]]
    assert deaths[0] == deaths[1] and np.isclose(deaths[0], 0.75*deaths[2], rtol=0.05) # Halved for the second half

    # A seasonal value peaks once a year, under every solver
    for solver in ['ode', 'binomial', 'gillespie', 'abm']:
        beta = ss.seasonal(mean=ss.perday(0.3), amplitude=0.5, peak='January')
        sim = ss.Sim(make(beta), start=2020, stop=2023, dt=ss.days(1), solver=solver, total_pop=20_000, **kw).run()
        n_I = sim.results.sis.I[366:731] # The second year
        assert np.argmax(n_I) < 45 or np.argmax(n_I) > 320, f'Expected infections to peak in the winter with {solver}'
        assert n_I.max() > 1.5*n_I.min()
    assert sim.unit == 'day'
    assert '0.3 per day × (1 + 0.5 cos(2π(t − 0.04167)))' in sim.describe(output=True) # Peaking in the middle of January

    # A branch (split on entry) whose rate varies in time is a wait at that rate, with its flow recorded (before, a compartmental run moved them all at once and recorded nothing)
    for solver in ['ode', 'binomial']:
        I, H, R = ss.states('I H R')
        severe = ss.Transition(I, H, p=0.25, rate=ss.par(0.2778, name='rate_severe')*ss.seasonal(mean=1.0, amplitude=0.3, peak=0.5))
        sir = ss.Disease(severe=severe, mild=ss.Transition(I, R, p=0.75, rate=0.0909), first='I', name='sir')
        res = ss.Sim(sir, total_pop=10_000, stop=60, dt=0.1, unit='day', solver=solver, verbose=0).run().results.sir
        assert res.H[0] == 0 and np.isclose(res.severe.cum[-1], res.H[-1], rtol=1e-3) # Was 2,500 at the start, and a flow of 0

    # The ODE takes a value that varies smoothly at the time of each stage of a step, so a step of a day is as good as a step of three hours; with a function in the model too
    for extra in [None, lambda sim: 0.0]:
        res = []
        for dt in [1, 1/8]:
            sis = make(ss.seasonal(mean=ss.perday(0.3), amplitude=0.5, peak='January'))
            if extra is not None:
                S, I = ss.states('S I')
                sis = ss.Disease(infection=ss.Transition(S, I, rate=sis.pars.beta.raw * I), recovery=ss.Transition(I, S, dur=10), death=ss.death(I, rate=extra), init_prev=0.01, name='sis')
            n = int(1/dt)
            res.append(ss.Sim(sis, start=2020, stop=2021, dt=ss.days(dt), total_pop=1e4, **kw).run().results.sis.I.values[::n]) # At the end of each day
        assert np.abs(res[0][:len(res[1])] - res[1]).max() < 1, 'Expected the ODE to take a seasonal value at each stage of a step'
    draws = ss.truncnorm(loc=2.5, scale=0.5, low=1.0, high=2.6, strict=False).rvs(1000)
    assert draws.min() >= 1 and draws.max() <= 2.6
    yearly = ss.timeseries({'2020-01-01': 1.0, '2020-07-01': 2.0}, period=ss.years(1))
    dated = ss.Sim(stop=ss.days(800), unit='day', start='2020-01-01', **kw).init()
    assert [yearly.at(year, dated) for year in [2020.1, 2020.6, 2021.1, 2021.6]] == [1, 2, 1, 2]
    curve = ss.timeseries({0: 1.0, 1: 0.0}, mode='cosine')
    assert np.isclose(curve.at(0.5, ss.Sim(stop=2, unit='year').init()), 0.5) and curve.at(0.1, ss.Sim(stop=2, unit='year').init()) > 0.9
    ts = ss.timeseries(['2010-01-01', '2010-07-01'], [0.5, 0.9]) # From its times and values, as np.interp takes them
    assert repr(ts) == repr(ss.timeseries({'2010-01-01': 0.5, '2010-07-01': 0.9}))
    assert ss.timeseries([0, 10], [1, 2], mode='linear').mode == ss.timeseries({0: 1, 10: 2}, 'linear').mode == 'linear'

    # A value that varies in time as a keyword of a module written as code, and a number times a time value
    seasonal = ss.seasonal(mean=ss.peryear(0.4), amplitude=0.6, peak='January')
    assert ss.Sim(ss.SIS(beta=seasonal), start=2000, stop=2010, n_agents=500, **kw).run().results.sis.I.max() > 0
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ 2*ss.seasonal(mean=0.15, amplitude=0.2, peak=0.1)*I.frac, recovery=I >> R @ ss.days(10), init_n=dict(I=10), name='sir')
    assert ss.Sim(sir, unit='day', stop=60, **kw).run().results.sir.I.max() > 100

    # An ODE notes where chance would matter: a start with a few infected, and a dip below one infected person
    S, I, R = ss.states('S I R')
    few = ss.Sim(ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_n=dict(I=3), name='few'), unit='day', stop=50, **kw).run()
    assert any('starts with 3 infected' in note for note in few.notes)
    S, I, R = ss.states('S I R')
    rounded = ss.Sim(ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_n=dict(I=10), name='few'), total_pop=1e6, n_agents=10_000, unit='day', stop=5, **kw).run()
    assert rounded.results.few.I[0] == 100 and any('rounds to 0' in note for note in rounded.notes) # As agents, 10 people are a tenth of an agent, so one agent (100 people) is seeded, which is said
    S, I = ss.states('S I')
    dip = ss.Disease(infection=ss.Transition(S, I, rate=ss.timeseries({0: 0.05, 60: 0.4}) * I), recovery=ss.Transition(I, S, dur=10), init_n=dict(I=20), name='dip')
    assert any('would usually die out' in note for note in ss.Sim(dip, unit='day', stop=200, **kw).run().notes)

    # A fast transition is noted when the solver takes whole steps, and check_dt() says how much the step matters
    S, I, R = ss.states('S I R')
    fast = ss.Disease(infection=ss.Transition(S, I, rate=0.6 * I), recovery=ss.Transition(I, R, dur=4), init_prev=0.01, name='fast')
    stepped = ss.Sim(fast, solver='binomial', substeps=1, unit='day', stop=60, total_pop=1e6, **kw) # Whole steps (by default, the chain binomial takes enough substeps)
    df = stepped.check_dt(output=True)
    assert any('22% leave in a step' in note for note in stepped.new().init().notes) # 1 - exp(-1/4)
    assert 0.05 < df.change[df.result == 'I'].iloc[0] < 0.2
    with sc.capture() as txt: # With its default substeps, the chain binomial takes substeps of the same size whatever the step, so says so (it printed a table of zeros)
        S, E, I, R = ss.states('S E I R')
        measles = ss.Disease(S >> E @ 1.9*I, E >> I @ ss.days(10), I >> R @ ss.days(8), init_n=dict(I=10), name='measles') # Substeps of half a day, at either step
        ss.Sim(measles, solver='binomial', stop=ss.days(60), total_pop=1e5, **kw).check_dt()
    assert 'does not change its results' in txt
    S, I, R = ss.states('S I R')
    slow = ss.Disease(infection=ss.Transition(S, I, rate=0.6 * I), recovery=ss.Transition(I, R, dur=4), init_prev=0.01, name='slow', dt=ss.days(2))
    own = ss.Sim(slow, solver='binomial', substeps=1, unit='day', dt=1, stop=60, total_pop=1e6, **kw).check_dt(output=True)
    assert own.change[own.result == 'I'].iloc[0] > 0.05 # A module with its own timestep takes a smaller one too
    smooth = ss.Sim(fast, solver='ode', unit='day', stop=60, total_pop=1e6, **kw)
    assert smooth.check_dt(output=True).change.max() < 1e-3 # An ODE does not depend on the timestep
    assert not smooth.new().init().notes
    quiet = ss.Sim(fast, solver='ode', stop=ss.days(60), verbose=0).init()
    assert any("unit='day'" in note for note in quiet.notes) # Kept, with nothing printed
    with sc.capture() as txt:
        ss.Sim(fast, networks=ss.RandomNet(), solver='ode', n_agents=1000, unit='day', stop=5, verbose=1).init()
    assert txt.index('as an ODE') < txt.index('RandomNet approximated') # The line that says how it is run comes before the notes it leads to
    assert 'n_agents=1000 is used as the population' in txt # A compartmental run has no agents, so it is the population

    # A rate times an ss.by holding a time series, as a variable, is the same as written in the rate (it was IncompatibleTypes); one inside an ss.by is refused when made (it failed at run time)
    school = ss.timeseries({'2000-01-01': 0.6, '2000-01-07': 1.15, '2000-04-01': 0.6}, period=ss.years(1))
    def school_run(make):
        S, E, I, R = ss.states('S E I R')
        m = ss.Disease(S >> E @ make()*I, E >> I @ ss.days(10), I >> R @ ss.days(8), init=dict(I=0.001), name='measles').stratify(age={0: 0.3, 15: 0.6, 65: 0.1})
        return ss.Sim(m, networks=ss.contacts([[12, 6, 0.5], [3, 8, 1], [1.5, 6, 3]]), start='2000-01-01', stop='2000-03-01', dt=ss.days(1), **kw).run().results.measles.S_to_E.values
    beta = ss.perday(0.125) * ss.by(age=[school, 1, 1])
    assert np.allclose(school_run(lambda: beta), school_run(lambda: ss.perday(0.125) * ss.by(age=[school, 1, 1])))
    with pytest.raises(TypeError, match='outside'):
        ss.by(age=[ss.perday(0.125)*school, ss.perday(0.125), ss.perday(0.125)])

    # A parameter inside a seasonal value needs no name: it is named after the value and its field (it was "Could not find a name")
    S, I, R = ss.states('S I R')
    beta = ss.seasonal(mean=ss.par(ss.perday(0.3), bounds=[ss.perday(0.1), ss.perday(1)]), amplitude=ss.par(0.2, bounds=[0, 0.5]))
    assert {'beta', 'beta_mean', 'beta_amplitude'} <= set(ss.Disease(S >> I @ beta*I, I >> R @ ss.days(8), name='sir').pars.keys())
    return sim


def test_stays_and_offspring():
    sc.heading('Testing a minimum stay (after=) and litters (ss.birth(per=)), as EMULSION has them')
    NB, W = ss.states(2)
    growth = ss.Disease(weaning=ss.Transition(NB, W, rate=ss.perday(0.5), after=ss.days(21)), first='NB', name='growth')
    sim = ss.Sim(growth, unit='day', stop=40, n_agents=5000, verbose=0).run()
    n_W = sim.results.growth.W.values
    assert sim.solver == 'abm' and n_W[21] == 0 and n_W[22] > 0.25*5000 # Nobody leaves for 21 days, then at the rate
    binomial = ss.Sim(growth, unit='day', stop=40, total_pop=5000, solver='binomial', verbose=0).run(n_runs=20).results.growth.W.values # As compartments, by a conveyor of whole steps
    ode = ss.Sim(growth, unit='day', stop=40, total_pop=5000, verbose=0).run()
    assert ode.solver == 'ode' and ode.results.growth.W[21] == 0 < ode.results.growth.W[22]
    assert np.max(binomial[:, 21]) == 0 and np.isclose(np.mean(binomial[:, 22]), n_W[22], rtol=0.05) # The same as agents, step for step
    E, D, I = ss.states(3) # An escape during a minimum stay, as EMULSION's: a rate only while the time in the state is short (agents)
    x = ss.Disease(onset=ss.Transition(E, I, rate=ss.perday(0.3), after=ss.days(10)), death=ss.Transition(E, D, rate=ss.perday(0.5)*(E.time < ss.days(10))), first='E', name='x')
    r = ss.Sim(x, unit='day', stop=30, n_agents=5000, verbose=0).run().results.x
    assert r.I[10] == 0 and r.I[11] > 0 and r.D[11] == r.D[30] > 0, 'nobody should become infectious before day 10, or die after it'
    S, I, R = ss.states('S I R') # A stay on a transmission is not built yet, and no engine ignores it
    for solver in ['abm', 'ode']:
        infection = ss.Transition(S, I, rate=0.3 * I)
        infection.after = ss.days(5)
        with pytest.raises(ss.CapabilityError):
            ss.Sim(ss.Disease(S, I, R, infection=infection, recovery=I >> R @ ss.days(10), init_n=dict(I=10), name='x'), n_agents=500, unit='day', stop=20, solver=solver, verbose=0).init()

    # A rate competing with a dwell of whole steps acts in each of them, including for people who arrived by a progression: 0.08 leave by it during a one-month dwell (not 0)
    S, X, A, B = ss.states('S X A B')
    d = ss.Disease(S, X, A, B, enter=ss.Transition(S, X, rate=ss.peryear(0.5)), timed=ss.Transition(X, A, dur=ss.constant(ss.months(1))), hazard=ss.Transition(X, B, rate=ss.peryear(1.0)), name='d')
    r = ss.Sim(d, n_agents=20_000, start=2000, stop=2010, dt=ss.months(1), verbose=0).run().results.d
    share = r.hazard.values.sum()/(r.hazard.values.sum() + r.timed.values.sum())
    assert np.isclose(share, 1 - np.exp(-1/12), rtol=0.1)

    G, L, NB = ss.states(3) # A litter is a birth driven by the transition (EMULSION's offspring)
    farrowing = ss.Transition(G, L, dur=ss.days(10))
    herd = ss.Disease(G, L, NB, farrowing=farrowing, weaning=ss.Transition(L, G, dur=ss.days(20)), litter=ss.birth(NB, per=farrowing, n=3), first='G', name='herd')
    res = ss.Sim(herd, unit='day', stop=60, n_agents=100, verbose=0).run().results.herd
    assert res.NB[-1] == 3*res.farrowing.values.sum() > 0 # Three newborns for each farrowing, who are people
    res = ss.Sim(herd, unit='day', stop=60, solver='ode', verbose=0).run().results.herd # And as compartments
    assert np.isclose(res.NB[-1], 3*res.farrowing.values.sum())

    # Offspring by the mother's state: a split on the birth, which fires on= the birth of either module (the one that makes the children, or one that also places them)
    for own in [False, True]:
        S, I = ss.states('S I')
        newborns = ss.birth(S, per=farrowing, n=3)
        trigger = newborns if own else herd.graph.transitions.litter
        health = ss.Disease(S, I, newborns=newborns, vertical=ss.Transition(S, [I, S], on=trigger, p=[0.5, 0.5], where=ss.mother(I, born=True)), init_n=dict(I=50), name='health')
        res = ss.Sim([herd, health], unit='day', stop=60, n_agents=100, verbose=0).run().results.health
        split = res['vertical.I'].values.sum() + res['vertical.S'].values.sum()
        assert 0 < split < res.newborns.values.sum() and res['vertical.I'].values.sum() > 0, f'on= {"its own" if own else "the herd\'s"} birth: {split} newborns split of {res.newborns.values.sum()}' # Those of infected mothers only
    return sim


def test_branch_where():
    """ A p= branch with where= has probability 0 outside it, and the other branches share the rest (as agents, men were sent to it and stayed for good; as compartments, a fixed delay with where= raised) """
    sc.heading('Testing where= on a branch')
    def make():
        S, A = ss.states('S A')
        return ss.Disease(infection=S >> A @ ss.perday(0)*A,
            persist = ss.Transition(A, S, p=0.9, dur=ss.constant(ss.years(100)), where=ss.ppl.female), # Women only
            clear_a = ss.Transition(A, S, p=None, dur=ss.days(30)), init_prev=0.5, name='d')
    strat = dict(sex=['female', 'male'])
    for solver in ['abm', 'ode']:
        size = dict(n_agents=4000) if solver == 'abm' else dict(total_pop=4000)
        r = ss.Sim(make(), solver=solver, stratify=strat, unit='day', stop=365, verbose=0, **size).run().results.d
        men = r.A.select(sex='male')
        women = r.A.select(sex='female')
        assert men[-1] < 0.01*men[0], f'{solver}: men should all clear (the remainder), not {men[-1]/men[0]:.2f} of them stay'
        assert np.isclose(women[-1], 0.9*women[0], rtol=0.05), f'{solver}: 90% of women persist, not {women[-1]/women[0]:.2f}'

    # A remainder for the same people as the branches, beside an exit for others (it raised)
    P, C, R = ss.states('P C R')
    female = lambda sim: sim.people.female.raw
    d = ss.Disease(P, C, R, to_cin=ss.Transition(P, C, dur=ss.years(3), p=0.25, where=female), to_clear=ss.Transition(P, R, dur=ss.years(1), where=female),
                   male=ss.Transition(P, R, dur=ss.years(1), where=lambda sim: ~sim.people.female.raw), first='P', name='d')
    res = ss.Sim(d, n_agents=10_000, start=2000, stop=2010, dt=ss.years(0.25), verbose=0).run().results.d
    assert 1000 < res.C[-1] < 1500 # A quarter of about 5,000 women
    return r


@sc.timer()
def test_branches():
    """ Immediate branches, and a branch that takes the remainder """
    sc.heading('Testing immediate branches and remainders')
    def make(which):
        S, E, P, A, R = ss.states('S E P A R')
        rest = dict(rec=ss.Transition(P, R, dur=5), rec2=ss.Transition(A, R, dur=5))
        if which == 'split': # A branch to each destination, with p= and no rate or duration: immediate
            trs = dict(sev=ss.Transition(E, P, p=0.3), mild=ss.Transition(E, A, p=0.7))
        else: # One branch with p=None and a duration gets the remainder (with no duration either, it would be undriven)
            trs = dict(sev=ss.Transition(E, P, p=0.3, dur=2), mild=ss.Transition(E, A, p=None, dur=2))
        return ss.Disease(infection=ss.Transition(S, E, rate=0.5 * (P + A)), **trs, **rest, init_prev=0.01, name='d')
    for which in ['split', 'timed']:
        for solver in ['ode', 'abm']:
            size = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
            sim = ss.Sim(make(which), solver=solver, unit='day', stop=60, **size, **kw).run()
            res = sim.results.d
            share = res.rec[1:].sum()/(res.rec[1:].sum() + res.rec2[1:].sum())
            assert np.isclose(share, 0.3, atol=0.05), f'{which} ({solver}): {share:.2f} went to P, not 0.3'
    with pytest.raises(ValueError, match='the branch probabilities out of state "E" sum to 0.3, not 1'): # Only one can take the remainder
        E, P, A, R = ss.states('E P A R')
        ss.Disease(sev=ss.Transition(E, P, p=0.3), a=ss.Transition(E, A, dur=2), b=ss.Transition(E, R, dur=2))
    with pytest.raises(ValueError, match='undriven, not the remainder'): # With no p= at all, a transition is undriven
        E, P, A = ss.states('E P A')
        ss.Disease(sev=ss.Transition(E, P, p=0.3), mild=ss.Transition(E, A))
    agents = dict(n_agents=1000, start=2000, stop=2003, dt=ss.years(0.25), **kw)
    S, J, A, B = ss.states('S J A B')
    with pytest.raises(ValueError, match='give it dur=0'): # An immediate split as one transition says how to write it (it crashed)
        ss.Sim(ss.Disease(go=S >> J @ ss.peryear(1), branch=ss.Transition(J, [A, B], p=[0.5, 0.5]), name='d'), **agents).init()
    A, B, C = ss.states('A B C')
    d = ss.Disease(A, B, C, tr=ss.Transition(A, [B, C], dur=ss.years(1), p=lambda sim, uids: np.tile([0.3, 0.7], (len(uids), 1))), first='A', name='d')
    with pytest.raises(ss.CapabilityError, match='one per destination'): # A row of probabilities from a function in p= is refused clearly (it crashed)
        ss.Sim(d, **agents).run()
    for solver in ['abm', 'ode']: # A branch probability over 1 raises, as a sum over 1 does at build (it was used as 1, killing 5,854 of 5,901)
        S, D, R = ss.states('S D R')
        d = ss.Disease(S, D, R, onset=S >> D @ ss.peryear(2), death=ss.Transition(D, ss.null, dur=ss.days(5), p=ss.by(type=[0.1, 1.5])), recovery=ss.Transition(D, R, dur=ss.days(5)), name='d')
        d.stratify(type=['watery', 'bloody'], states=[D])
        with pytest.raises(ValueError, match='1.5'):
            ss.Sim(d, n_agents=1000, solver=solver, start=2000, stop=2001, **kw).run()

    # A branch probability set after building: a new value for one that was given, and a clear error for one that was not (it raised "'by' object has no attribute 'leaves'")
    S, I, R = ss.states('S I R')
    d = ss.Disease(S, I, R, infection=S >> I @ ss.peryear(1), death=ss.Transition(I, ss.null, dur=ss.days(20), p=0.1), rec=ss.Transition(I, R, dur=ss.days(20)), name='d')
    d.death.p = ss.by(age=[0.5, 0.0])
    d.stratify(age=[0, 15])
    res = ss.Sim(d, n_agents=5000, use_aging=False, start=2000, stop=2001, **kw).run().results.d
    assert res.death.by('age').values.sum(0)[1] == 0 and res.death.sum() > 0
    d = ss.Disease(S, I, infection=S >> I @ ss.peryear(0.1), death=I >> ss.null @ ss.days(20), name='d')
    with pytest.raises(ValueError, match='when the transition is made'):
        d.death.p = ss.by(age=[0.1, 0.01])

    # Branches kept to one level each by move= sum to 1 within that level, not across levels (FPsim review: p= per parity)
    A, B, C = ss.states('A B C')
    trs = {f'{dst.name.lower()}{k}': ss.Transition(A, dst, p=p, dur=ss.months(1), move=dict(k=(k, '1'))) for dst,p in [(B, 0.9), (C, 0.1)] for k in '01'}
    d = ss.Disease(A, B, C, **trs, first='A', name='d').stratify(k=['0', '1']) # Everyone starts in A
    res = ss.Sim(d, n_agents=2000, start=2000, stop=2001, dt=ss.months(1), unit='year', **kw).run().results.d
    assert res.A[-1] == 0 and np.isclose(res.C[-1]/2000, 0.1, atol=0.03)
    A, B, C = ss.states('A B C')
    with pytest.raises(ValueError, match=r'1.8.*\(b1: .*c1: '): # But still within each
        ss.Disease(A, B, C, b0=ss.Transition(A, B, p=0.9, dur=1, move=dict(k=('0', '1'))), c0=ss.Transition(A, C, p=0.1, dur=1, move=dict(k=('0', '1'))),
                   b1=ss.Transition(A, B, p=0.9, dur=1, move=dict(k=('1', '1'))), c1=ss.Transition(A, C, p=0.9, dur=1, move=dict(k=('1', '1'))))
    return sim


def test_source_weights():
    sc.heading('Testing weights on the sources of infection, beta * (P + 0.55*EL), with beta a value or a parameter')
    def make(form, beta):
        S, P, EL = ss.states('S P EL')
        stages = dict(latent=ss.Transition(P, EL, dur=ss.years(1)), clear=ss.Transition(EL, S, dur=ss.years(2)))
        if form == 'dict':
            return ss.Disease(infection=ss.Transition(S, P, rate=beta * (P + 0.55*EL)), **stages, init=dict(P=0.05), name='syph')
        return ss.Disease(S, P, EL, infection=S >> P @ ss.par(beta, name='beta') * (P + 0.55*EL), **stages, init=dict(P=0.05), name='syph')
    for beta, net, solver in [(ss.peryear(0.5), ss.MixingPool(n_contacts=4), 'ode'), (ss.peryear(0.5), ss.MixingPool(n_contacts=4), 'abm'), (dict(sexualnet=ss.peract(m2f=0.05, f2m=0.03)), ss.SexualNet(), 'abm')]: # A dict or ss.peract times a weighted sum (a TypeError before)
        res = [ss.Sim(make(form, beta), networks=net, n_agents=3000, total_pop=3000, start=2000, stop=2010, dt=ss.months(1), solver=solver, verbose=0).run().results.syph.P.values for form in ['dict', 'op']]
        assert np.array_equal(res[0], res[1]) and res[0][-1] > 0, f'{solver}: {res[0][-1]} against {res[1][-1]}'
    assert 'from P + 0.55 EL' in make('dict', 0.3).describe(output=True)
    return


def test_cond_by_expr():
    sc.heading('Testing arithmetic on a condition and an ss.by: an expression for a rate, in either order')
    age = ss.by(age=[0.1, 0.3])
    older = ss.ppl.age >= 50
    forms = {
        'by + cond':  lambda: age + 0.2*older,
        'cond + by':  lambda: 0.2*older + age,
        'by * cond':  lambda: age * (1 + older),
        'cond * by':  lambda: (1 + older) * age,
        'cond - by':  lambda: 1 + older - age,
        'exp(cond) * by': lambda: np.exp(np.log(2)*older) * age,
        'par(by) + cond': lambda: ss.par(age, name='agefx') + 0.2*older,
        'cond + par(by)': lambda: 0.2*older + ss.par(age, name='agefx'),
    }
    rates = {'by + cond':[0.1, 0.5], 'cond + by':[0.1, 0.5], 'by * cond':[0.1, 0.6], 'cond * by':[0.1, 0.6], 'cond - by':[0.9, 1.7],
             'exp(cond) * by':[0.1, 0.6], 'par(by) + cond':[0.1, 0.5], 'cond + par(by)':[0.1, 0.5]} # Under 50, then 50 and over
    def make(form):
        A, B = ss.states('A B')
        return ss.Disease(A, B, go=A >> B @ ss.peryear(1.0)*forms[form](), first='A', name='c').stratify(age=[0, 50])
    pars = dict(start=2000, stop=2002, dt=ss.months(1), verbose=0)
    for form in forms:
        expected = 1 - np.exp(-2*np.array(rates[form])) # The share of each age group that moves in two years
        for solver,size in [('ode', dict(total_pop=10_000)), ('abm', dict(n_agents=10_000))]:
            res = ss.Sim(make(form), solver=solver, **size, **pars).run().results.c
            moved = res.go.by('age').values.sum(axis=0)/res.A.by('age').values[0]
            assert np.allclose(moved, expected, rtol=0.02 if solver == 'ode' else 0.1), f'{form} ({solver}): {moved} moved, not {expected}'

    # A bare ss.by in a rate with a property of people that is not a dimension, as agents (the reported case)
    by_year = ss.by(dict(age=list(range(100)), value=np.linspace(0, 1, 100)))
    for rate in [by_year + 0.5*ss.ppl.female, 0.5*ss.ppl.female + by_year, by_year * (1 + ss.ppl.female), (1 + ss.ppl.female) * by_year]:
        assert isinstance(rate, ss.expressions.Expr), f'{rate!r} should be an expression'
        A, B = ss.states('A B')
        d = ss.Disease(A, B, go=A >> B @ ss.peryear(1.0)*rate, first='A', name='c')
        assert ss.Sim(d, n_agents=500, **pars).run().results.c.go.sum() > 0

    # An unnamed ss.par inside an expression is an error when the model is built, not when it runs
    A, B = ss.states('A B')
    with pytest.raises(ValueError, match='Could not find a name'):
        ss.Disease(A, B, go=A >> B @ ss.peryear(1.0)*(ss.par(age) + 0.2*older), first='A', name='c')

    # np.clip of arithmetic on a condition (a state of another module, by name) is an expression, the same as np.maximum and np.minimum
    def clipped(fn):
        A, B, cell, other = ss.states('A B cell other')
        return [ss.Disease(cell, other, first='cell', name='b'), ss.Disease(A, B, go=A >> B @ ss.peryear(1.0)*fn(3*ss.state('b.cell')), first='A', name='c')]
    res = [ss.Sim(clipped(fn), n_agents=2000, **pars).run().results.c.go.values for fn in [lambda x: np.clip(x, 0, 1), lambda x: np.minimum(np.maximum(x, 0), 1)]]
    assert np.array_equal(res[0], res[1]) and np.isclose(res[0].sum()/2000, 1 - np.exp(-2), rtol=0.05)
    return


@sc.timer()
def test_modify_dist_dwell():
    sc.heading('Testing modifiers on a dwell-time parameter that is a distribution')
    def make(raw=ss.constant(ss.years(1)), mod=None, solver='abm'):
        A, B = ss.states('A B')
        dis = ss.Disease(A, B, go=ss.Transition(A, B, rate=ss.peryear(2)), back=ss.Transition(B, A, dur=ss.par(raw, name='d')), first='A', name='c')
        size = dict(n_agents=2000) if solver == 'abm' else dict(total_pop=2000)
        return ss.Sim(dis, interventions=mod, start=2000, stop=2006, dt=ss.years(0.25), solver=solver, rand_seed=1, **size, **kw)

    # After the start, everyone entering B leaves it after the new dwell time, 5 years, by set=, multiply= or add=; before it, after 1 year
    def dwells(sim):
        c = sim.diseases.c
        entered, left = c.ti_B.raw.astype(float), c.ti_exit_B.raw.astype(float)
        dwell = (left - entered)*sim.t.dt.value # In years
        start = 4 # 2001
        inB = c.B.raw & sim.people.alive.raw
        return dwell[inB & (entered >= start)], dwell[inB & (entered < start)]
    for kwargs in [dict(set=ss.constant(ss.years(5))), dict(multiply=5), dict(add=ss.years(4))]:
        sim = make(mod=ss.modify('c.pars.d', start='2001', **kwargs)).run()
        after, _ = dwells(sim)
        assert len(after) > 100 and np.allclose(after, 5, atol=0.25), f'{kwargs}: dwell times of later entrants {np.unique(after)}'
        assert sim.results.c.back[-12:].sum() < 0.5*make().run().results.c.back[-12:].sum()
    _, before = dwells(make().run())
    assert np.allclose(before, 1, atol=0.25)

    # set= with a duration replaces the distribution with an exponential of that mean, as dur= of a transition does
    sim = make(mod=ss.modify('c.pars.d', set=ss.years(5), start='2001')).run()
    after, _ = dwells(sim)
    assert after.std() > 2 and 4 < after.mean() < 12 # Of those still in B, so longer than the mean

    # What cannot change raises: as compartments, a distribution laid out at the start; a distribution set on a duration that is a mean; set= of a distribution for only some people
    with pytest.raises(ss.CapabilityError, match='solver="abm"'):
        make(mod=ss.modify('c.pars.d', set=ss.constant(ss.years(5)), start='2001'), solver='ode').init()
    with pytest.raises(ss.CapabilityError, match='solver="abm"'):
        make(mod=ss.modify('c.pars.d', multiply=5, start='2001'), solver='binomial').init()
    with pytest.raises(ValueError, match='distribution'):
        make(raw=ss.years(1), mod=ss.modify('c.pars.d', set=ss.constant(ss.years(5)), start='2001')).init()
    with pytest.raises(ss.CapabilityError, match='where='):
        make(mod=ss.modify('c.pars.d', set=ss.constant(ss.years(5)), where=lambda sim: sim.people.age > 30)).init()
    return


def test_algebra():
    """ Each kind of node in an expression, under each operator and in each slot, in each solver: it runs, or it raises a clear error """
    sc.heading('Testing the expression algebra over the solvers...')
    solvers = ['abm', 'ode', 'binomial']
    agents_only = lambda solver: solver != 'abm' # Raises a CapabilityError under the compartmental solvers
    def run(solver, *diseases, stratify=dict(age=[0, 15, 50], sex=['female', 'male']), **kwargs):
        sim = ss.Sim(diseases=list(diseases), solver=solver, n_agents=200, total_pop=200, start=2000, stop=2002, dt=ss.months(2), rand_seed=1, verbose=0, **kwargs)
        sim.stratify(**stratify)
        return sim.run()
    def check(label, solver, func, fails=False):
        if fails:
            with pytest.raises(ss.CapabilityError, match='solver'):
                func(solver)
        else:
            sim = func(solver)
            assert all(np.all(np.isfinite(res.values)) for res in sim.results.flatten().values() if isinstance(res, ss.Result)), f'{label} ({solver}): not finite'
        return

    # Each kind of node, as a factor in a rate, under each operator with an ss.by table
    table = lambda: ss.by(dict(age=[0, 15, 50], value=[0.8, 1.0, 1.2]))
    nodes = dict(
        number   = lambda I: ss.par(0.9, name='k'),
        table    = lambda I: ss.by(dict(age=[0, 15, 50], value=[0.5, 1.0, 0.8])),
        keyed    = lambda I: ss.by(sex=dict(female=1.1, male=0.9)),
        count    = lambda I: 1 + I.frac,
        person   = lambda I: ss.ppl.age/50,
        attr     = lambda I: ss.attr(ss.uniform(0.5, 1.5), name='a'),
        series   = lambda I: ss.timeseries({2000: 1.0, 2001: 0.5}),
        seasonal = lambda I: ss.seasonal(amplitude=0.3),
    )
    ops = {
        '+': lambda x: x + table(),
        '-': lambda x: 2 - x*table(),
        '*': lambda x: table()*x,
        '/': lambda x: x/table(),
        '**': lambda x: x**2,
        'minimum': lambda x: np.minimum(x, table()),
        'exp': lambda x: np.exp(-x),
    }
    def node_model(node, op):
        def make(solver):
            S, I = ss.states('S I')
            return run(solver, ss.Disease(infection=S >> I @ ss.peryear(2)*op(node(I))*I, recovery=I >> S @ ss.years(0.5), init=dict(I=0.1), name='sis'))
        return make
    for nname,node in nodes.items():
        for oname,op in ops.items():
            for solver in solvers:
                check(f'{nname} {oname}', solver, node_model(node, op)) # A trait (an attribute drawn once) at its mean under a compartmental solver

    # Each slot that takes an expression, with the forms the reviews found
    def sis(rate=None, recovery=None, **kwargs):
        S, I = ss.states('S I')
        rate = ss.peryear(2)*I if rate is None else rate(I)
        recovery = I >> S @ ss.years(0.5) if recovery is None else recovery(I, S)
        return ss.Disease(infection=S >> I @ rate, recovery=recovery, init=dict(I=0.1), name='sis', **kwargs)
    def factor_peract(solver): # A number times a per-act beta, with values in years here and in months in the other disease
        S, I = ss.states('S I')
        other = ss.Disease(S, I, infection=S >> I @ ss.peract(0.02)*I, recovery=I >> S @ ss.months(6), init=dict(I=0.1), name='other')
        return run(solver, sis(rate=lambda I: 0.3*(dict(sexualnet=ss.peract(0.05))*I)), other, networks=ss.SexualNet())
    def result_ratio(solver): # A flow times a number over a sum of states, by group
        S, I, R = ss.states('S I R')
        infection = S >> I @ ss.peryear(2)*I
        sim = run(solver, ss.Disease(S, I, R, infection=infection, recovery=I >> R @ ss.years(0.5), rate=1e5*infection/(S + I + R), init=dict(I=0.05), name='sir'))
        assert len(sim.results.sir.rate.select(sex='female')) == len(sim.results.sir.rate) # Kept by group
        return sim
    def family_ratio(solver): # A member's flow over the family's, by group
        def member(name, beta):
            S, I, X = ss.states('S I X')
            return ss.Disease(S, I, X, infection=S >> I @ ss.peryear(beta)*I, cancer=I >> X @ ss.years(2), init=dict(I=0.05), name=name)
        sim = run(solver, ss.Disease(name='hpv', strains=ss.Strains([member('hpv16', 2), member('hpv18', 1.5)])), stratify=dict(age=[0, 25, 50]))
        share = sim.results.hpv16.cancer/sim.results.hpv.cancer
        vals = share.values[1:][~np.isnan(share.values[1:])] # A step with no cancers in the family is 0/0
        assert np.all((vals >= 0) & (vals <= 1))
        return sim
    def person_amount(solver): # A modifier's amount over people's properties
        d = sis()
        return run(solver, d, connectors=ss.modify(d.infection, multiply=ss.ppl.age/50))
    def other_states(solver): # A modifier's amount over another module's states
        A, B = ss.states('A B')
        d = sis()
        m = ss.Disease(A, B, on=A >> B @ ss.peryear(0.5), first='A', name='m')
        return run(solver, d, m, connectors=ss.modify(d.infection, multiply=1 - 0.9*m.B))
    def attr_on_exit(solver): # An attribute set as people leave a state
        S, I = ss.states('S I')
        return run(solver, ss.Disease(infection=S >> I @ ss.peryear(2)*I, recovery=I >> S @ ss.years(0.5), t_inf=ss.attr(I.time, on=I.exit), init=dict(I=0.1), name='sis'))
    def dwell_births(solver): # A dwell time over people's properties and clocks, once people are born
        S, I = ss.states('S I')
        dur = np.maximum(6 - ss.ppl.age, 0.5)*ss.years(1) + I.time
        return run(solver, ss.Disease(infection=S >> I @ ss.peryear(2)*I, recovery=ss.Transition(I, S, dur=dur), init=dict(I=0.1), name='sis'), demographics=[ss.birth(rate=ss.peryear(0.05)), ss.death(rate=ss.peryear(0.02))])
    def peract_dict(solver): # A per-act beta for each direction, in a dict by route
        return run(solver, sis(rate=lambda I: dict(sexualnet=ss.peract(m2f=0.05, f2m=0.03))*I), networks=ss.SexualNet())
    def by_contacts(solver): # A value by sex over a contact matrix by sex
        return run(solver, sis(rate=lambda I: ss.by(sex=dict(female=0.3, male=0.2))*ss.peryear(10)*I), networks=ss.contacts([[0, 1], [1, 0]], dims='sex'))
    def route_sources(solver): # A rate for each route, each with its own source weights
        S, A, B = ss.states('S A B')
        rate = dict(random=ss.peryear(1)*(A + 0.5*B), home=ss.peryear(2)*(A + B))
        d = ss.Disease(infection=S >> A @ rate, prog=A >> B @ ss.years(1), rec=B >> S @ ss.years(1), init=dict(A=0.1), name='d')
        return run(solver, d, networks=[ss.RandomNet(), ss.RandomNet(name='home')])
    slots = [factor_peract, result_ratio, family_ratio, person_amount, other_states, attr_on_exit, dwell_births, peract_dict, by_contacts, route_sources]
    agents = [attr_on_exit, dwell_births, route_sources] # Which only agents can run (other_states runs as compartments too, with the two diseases tracked jointly)
    for func in slots:
        for solver in solvers:
            check(func.__name__, solver, func, fails=func in agents and agents_only(solver))
    return


def test_init_expression():
    """ TY2: init= takes an expression of people's properties and parameters, under every solver, and its parameters can be set by name """
    class M(ss.Disease):
        A = ss.par(20.0)
        k = ss.par(1.0)
        age = ss.ppl.age # A helper, not a result
        U = ss.State()
        S = ss.State(init=np.minimum(1, 1 - (A - age)/(k*age + A))) # Gauld et al.'s share susceptible by age
        first = 'U'
        become = U >> S @ ss.peryear(0.01)
    share = lambda sim: sim.results.m.S[0]/(sim.results.m.S[0] + sim.results.m.U[0])
    sim = ss.Sim(diseases=M(), n_agents=20_000, dur=ss.days(3), dt=ss.days(1), verbose=0, rand_seed=1).run()
    age = sim.people.age.raw[sim.people.auids]
    expected = np.mean(np.minimum(1, 1 - (20 - age)/(age + 20)))
    assert abs(share(sim) - expected) < 0.01
    ode = ss.Sim(diseases=M(), n_agents=20_000, dur=ss.days(3), dt=ss.days(1), verbose=0, solver='ode', use_aging=True)
    ode.stratify(age=[0, 5, 10, 15, 20, 100])
    assert abs(share(ode.run()) - expected) < 0.03 # By age group
    fitted = ss.Sim(diseases=M(), n_agents=20_000, dur=ss.days(3), dt=ss.days(1), verbose=0, rand_seed=1, pars={'m.A': 5}).run()
    assert share(fitted) > share(sim) + 0.05
    class Q(ss.Disease): # Given in __init__(), reading the class's parameters
        U, S = ss.states('U S')
        first = 'U'
        A = ss.par(20.0)
        become = U >> S @ ss.peryear(0.01)
        def __init__(self, **kwargs):
            super().__init__(init=dict(S=np.minimum(1, ss.ppl.age/self.A)), **kwargs)
    q = ss.Sim(diseases=Q(), n_agents=5000, dur=ss.days(3), dt=ss.days(1), verbose=0).run()
    assert list(q.diseases.q.pars.keys()).count('A') == 1 and 0.3 < q.results.q.S[0]/(q.results.q.S[0] + q.results.q.U[0]) < 1
    return sim


def test_class_helpers():
    """ TY4, MS11: an expression per person as a class member is a helper, not a result; another name for a parameter is the same parameter; a parameter given as a result under its own name is an error """
    class T(ss.Disease):
        S, I = ss.states('S I')
        p_gall = ss.par(ss.by(age={0: 0.1, 30: 0.2}))
        p_cpg = ss.par(0.15)
        A = ss.par(20.0)
        A2, age = A, ss.ppl.age # Another name, and a property of people
        p_carrier = p_cpg*p_gall # A value for each person (by age)
        rel = age/A2
        rec = ss.Transition(I, [S, ss.null], p=[p_carrier, None], dur=ss.days(3))
        inf = S >> I @ ss.perday(0.01)*rel*I
        prevalence = I/(S + I)
    sim = ss.Sim(diseases=T(init_prev=0.1), n_agents=500, dur=ss.days(10), dt=ss.days(1), verbose=0).run()
    res, pars = sim.results.t, sim.diseases.t.pars
    assert 'prevalence' in res and not any(k in res for k in ['p_carrier', 'rel'])
    assert 'A' in pars and 'A2' not in pars
    t = ss.Sim(diseases=T(A=30.0, init_prev=0.1), n_agents=100, dur=ss.days(3), dt=ss.days(1), verbose=0).init().diseases.t # A value given to a parameter that has another name in the class (Typhoid's A = unexp2sus_saturation_age)
    assert 1*t.pars.A == 30 and 'A2' not in t.pars
    eir = ss.par(ss.perday(0.01), name='eir')
    S, I = ss.states('S I')
    with pytest.raises(ValueError, match='a name of its own'):
        ss.Disease(S >> I @ eir, I >> S @ ss.perday(0.1), eir=eir, name='m')
    S, I = ss.states('S I')
    m = ss.Disease(S >> I @ eir, I >> S @ ss.perday(0.1), eir_now=1*eir, name='m') # As the docs say: a result of a parameter is an expression
    assert 'eir_now' in ss.Sim(diseases=m, n_agents=100, dur=ss.days(3), dt=ss.days(1), verbose=0).run().results.m
    return sim


def test_describe_init_func():
    """ MS10: describe() names a function that gives an initial value """
    class M(ss.Disease):
        S, I = ss.states('S I')
        inf = S >> I @ ss.peryear(0.1)
        def start(self, sim):
            return 0.1
        def __init__(self, **kwargs):
            super().__init__(init=dict(I=self.start), **kwargs)
    sim = ss.Sim(diseases=M(), n_agents=100, verbose=0).init()
    assert 'I = start(sim)' in str(sim.diseases.m.describe(output=True))
    return


def test_counter_in_expression():
    """ TY14: a counter's level is read in a rate as a property of people, as agents and as compartments """
    out = {}
    for solver in ['abm', 'ode']:
        S, I = ss.states('S I')
        tppi = ss.par(0.5)
        infection = ss.Transition(S, I, rate=(1 - tppi)**ss.ppl.infections*ss.perday(0.3)*I, increment='infections')
        sis = ss.Disease(infection, I >> S @ ss.days(7), init=dict(I=0.01), name='sis')
        sis.stratify(infections=list(range(6)))
        sim = ss.Sim(diseases=sis, n_agents=10_000, dur=ss.days(150), dt=ss.days(1), verbose=0, solver=solver, rand_seed=1).run()
        out[solver] = sim.results.sis.infection.sum()/10_000
    assert abs(out['abm'] - out['ode']) < 0.1*out['ode'] and out['ode'] < 3 # Fewer than with no protection (about 5 infections per person)
    return out


def test_inflow_add():
    """ TY15: ss.modify(inflow, add=) adds an amount from outside the model to an inflow from ss.null, under every solver """
    for solver in ['abm', 'ode']:
        S, I = ss.states('S I')
        cfu = ss.State(people=False)
        ty = ss.Disease(S >> I @ ss.perday(1e-9)*cfu.n, I >> S @ ss.days(10), shedding=ss.null >> cfu @ ss.perday(100)*I.n, decay=cfu >> ss.null @ ss.perday(0.5), init=dict(I=0.0), name='ty')
        added = ss.modify('ty.shedding', add=ss.perday(1e6), start=2000.02)
        sim = ss.Sim(diseases=ty, interventions=added, n_agents=1000, start=2000, dur=ss.days(40), dt=ss.days(1), verbose=0, solver=solver).run()
        cfus = sim.results.ty.cfu.values
        assert cfus[3] == 0 and 1.9e6 < cfus[-2] < 2.2e6 # Added from day 8, decaying at half a day: about 2 million at equilibrium
    with pytest.raises(ValueError, match='inflow'):
        S, I = ss.states('S I')
        sir = ss.Disease(S >> I @ ss.perday(0.1)*I, name='sir')
        ss.Sim(diseases=sir, interventions=ss.modify('sir.S_to_I', add=1), n_agents=100, verbose=0).init() # Not an inflow
    return sim


def test_gated_function():
    """ FP12: a function rate times a condition is called only for the agents the condition holds for, and gives the same flows as the function taking care of it itself """
    sc.heading('A function rate gated by a condition')
    asked = []
    def young_rate(sim, uids):
        asked.append(np.asarray(sim.people.age[uids]))
        return ss.peryear(np.full(len(uids), 0.2))
    def young_rate_itself(sim, uids):
        return ss.peryear(np.where(sim.people.age[uids] < 20, 0.2, 0.0))
    def make(rate):
        S, I = ss.states('S I')
        return ss.Disease(dying=S >> ss.null @ rate, recovery=I >> S @ ss.years(1), name='d')
    gated = ss.Sim(diseases=make(ss.par(young_rate, name="young")*(ss.ppl.age < 20)), n_agents=2000, start=2000, stop=2005, rand_seed=3, **kw).run()
    itself = ss.Sim(diseases=make(young_rate_itself), n_agents=2000, start=2000, stop=2005, rand_seed=3, **kw).run()
    seen = np.concatenate(asked)
    assert len(seen) and np.all(seen < 20), 'The function is called only for those under 20'
    assert np.array_equal(gated.results.d.dying.values, itself.results.d.dying.values), 'The same deaths as a function that gives 0 to the others'

    # ss.Pregnancy's infant_rate, a function of the children it is asked about
    ages = []
    def infant_rate(sim, uids):
        ages.append(np.asarray(sim.people.age[uids]))
        return ss.peryear(np.full(len(uids), 0.05))
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(100), infant_rate=infant_rate)
    sim = ss.Sim(demographics=preg, n_agents=2000, start=2000, stop=2004, dt=ss.months(1), **kw).run()
    ages = np.concatenate(ages)
    assert len(ages) and np.all((ages >= 0) & (ages < 1)), 'infant_rate is called only for the children under 1'
    assert sim.results.pregnancy.infant_mortality.sum() > 0
    return sim


def test_compiled_eval():
    """ Agents evaluate a rate from a function made once (by group where a part is the same within a group, with shared parts once, and powers from a table): the same numbers as evaluating it directly """
    sc.heading('Compiled expressions as agents')
    S, I = ss.states('S I')
    k = ss.par(ss.by(patch=[1.0, 2.0, 0.5]), name='k')
    n_inf = ss.attr(1, on=I.enter, combine='sum', name='n_inf')
    d = ss.Disease(infection=S >> I @ ss.peryear(0.4)*k*(0.5**n_inf)*I.n/(S + I).n, recovery=I >> S @ ss.years(0.5), init=dict(I=0.1), name='d')
    sim = ss.Sim(diseases=d, n_agents=3000, start=2000, stop=2003, rand_seed=1, **kw)
    sim.stratify(patch=ss.geo.geography(names=['A', 'B', 'C'], pop=[1000, 1000, 1000]))
    sim.run()
    engine = sim.diseases.d.engine
    rate = sim.diseases.d.graph.transitions.infection.rate
    uids = sim.people.auids
    compiled = engine.eval_for(rate, uids)
    with engine.evaluating(uids=uids, for_uids=True):
        direct = ss.expressions.evaluate(rate, engine)
    assert np.array_equal(np.broadcast_to(compiled, len(uids)), np.broadcast_to(direct, len(uids))), 'The same numbers'
    assert np.ptp(compiled) > 0, 'Different by patch and by the number of infections'
    power = ss.solvers.agents.power
    whole = np.array([0, 1, 2, 3, 0, 7]*50, dtype=float)
    assert np.array_equal(power(0.02, whole), 0.02**whole), 'Powers of whole numbers from a table'
    part = whole + 0.5
    assert np.array_equal(power(0.02, part), 0.02**part)
    return sim


@sc.timer()
def test_certain_and_substeps():
    sc.heading('Testing a probability of 1 under every solver (FP1), and the chain binomial\'s default substeps near threshold (TB3)')
    # Everyone moves in the step, shared equally between two certain exits; a finite rate competing with them takes nobody
    S, I, A, B, D = ss.states('S I A B D')
    def make():
        return ss.Disease(S, I, A, B, D, infection=S >> I @ ss.peryear(0.5), toa=I >> A @ ss.probperyear(1), tob=I >> B @ ss.probperyear(1)*0.5, die=I >> D @ ss.peryear(2), init=dict(I=0.5), name='d')
    for solver in ['abm', 'ode', 'binomial', 'tau', 'gillespie']:
        res = ss.Sim(make(), n_agents=2000, solver=solver, start=2000, stop=2001, dt=ss.years(0.1), **kw).run().results.d
        first = res.I[0]
        assert res.D[1] <= 0.01*first, solver # In compartments, those infected within a substep wait for the next, and a few die first
        assert np.isclose(res.A[1] + res.B[1], first + (res.S[0] - res.S[1]), rtol=0.05) and np.isclose(res.A[1], res.B[1], rtol=0.15), (solver, res.A[1], res.B[1])
    with pytest.raises(ValueError, match='infinite number of arrivals'):
        ss.Sim(ss.Disease(S, I, birth=ss.birth(S, rate=ss.probperyear(1)), name='d'), n_agents=100, solver='ode', start=2000, stop=2001, **kw).run()

    # An SIS just above threshold: the default substeps follow the ODE (whole steps gave 0.083 for 0.048)
    def sis():
        S, I = ss.states('S I')
        return ss.Disease(S, I, infection=S >> I @ 1.05*I.frac, recovery=I >> S @ 1.0, init=dict(I=0.1), name='d')
    pars = dict(unit='year', dt=ss.months(1), start=0, stop=50, total_pop=1e9, **kw)
    ode = ss.Sim(sis(), solver='ode', **pars).run().results.d.I[-1]
    sim = ss.Sim(sis(), solver='binomial', **pars).run()
    assert sim.diseases.d.engine.substeps() >= 15 and np.isclose(sim.results.d.I[-1], ode, rtol=0.08), (sim.diseases.d.engine.substeps(), sim.results.d.I[-1]/ode)
    return sim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    count = test_counts()
    birth = test_births()
    strat = test_stratify()
    fit   = test_outcomes()
    comp  = test_likelihood()
    msim  = test_modify()
    funcs = test_functions()
    times = test_time_values()
    stays = test_stays_and_offspring()
    ddist = test_modify_dist_dwell()
    weights = test_source_weights()
    condby = test_cond_by_expr()
    branch = test_branches()
    bwhere = test_branch_where()
    test_algebra()
    test_init_expression()
    test_class_helpers()
    test_describe_init_func()
    test_counter_in_expression()
    test_inflow_add()
    gated = test_gated_function()
    compiled = test_compiled_eval()
    certain = test_certain_and_substeps()
