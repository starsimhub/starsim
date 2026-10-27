"""
Test dimensions that are more than a property of people: states with dimensions of their own, strains, and places
"""

# %% Imports and settings
import sys
import math
import subprocess
import tempfile
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sps
import scipy.integrate as sp_integrate
import scipy.stats as stats
from scipy.special import expit
import sciris as sc
import starsim as ss
import corpus

kw = dict(verbose=0)
solvers = ['ode', 'binomial', 'abm']


def at_start(sim):
    """ The results with the number in each state before the first step at index 0, for a compartmental sim (results count the number after each step) """
    sim.init()
    for mod in sim.declared:
        mod.engine.start_step()
        mod.engine.record_stocks()
        mod.engine.update_results()
    return sim.results


def run(mod, solver, **kwargs):
    """ Run a module with one solver, as 10,000 agents if it is agent-based """
    pars = dict(n_agents=10_000) if solver == 'abm' else {}
    return ss.Sim(mod, solver=solver, **pars, **kw, **kwargs).run()


def make_malaria():
    """ Ross-Macdonald: humans and mosquitoes as states of one disease """
    a = 0.3 # Bites per mosquito per day
    b = 0.5 # Probability an infectious bite infects a human
    c = 0.5 # Probability a bite on an infectious human infects the mosquito
    Sh, Eh, Ih, Rh = ss.states(4)
    Sv, Ev, Iv     = ss.states(3, people=False)
    Nh             = (Sh + Eh + Ih + Rh).n
    malaria = ss.Disease(
        infection  = Sh >> Eh @ a * b * Iv.n / Nh,
        latency    = Eh >> Ih @ ss.days(12),
        recovery   = Ih >> Rh @ ss.days(100),
        waning     = Rh >> Sh @ ss.days(365),
        emergence  = ss.null >> Sv @ ss.perday(5_000),
        biting     = Sv >> Ev @ a * c * Ih.n / Nh,
        incubation = Ev >> Iv @ ss.days(10),
        mortality  = Sv + Ev + Iv >> ss.null @ ss.days(10),
        init       = dict(Ih=0.01, Sv=50_000),
        first      = 'Sh',
        name       = 'malaria',
    )
    return malaria


def make_tb():
    """ TB with drug resistance as a strain, acquired during treatment """
    susceptible, latent, active, on_treatment = ss.states(4)
    tb = ss.Disease(
        infection  = susceptible >> latent @ ss.peryear(8) * active,
        activation = latent >> active @ ss.peryear(0.05),
        diagnosis  = active >> on_treatment @ ss.peryear(0.7),
        cure       = on_treatment >> susceptible @ ss.months(6),
        failure    = on_treatment >> active @ ss.peryear(0.2),
        init       = dict(active=0.01),
        name       = 'tb',
        strains    = ss.Strains(['DS', 'MDR'], name='resistance', split=[1.0, 0.0], infection=[1.0, 0.7], cure=[1.0, 0.3]),
    )
    tb.stratify(acquired=ss.transfer(resistance=('DS', 'MDR'), rate=ss.peryear(0.02), states=['on_treatment']))
    return tb


# %% Define the tests


def make_resistance(beta=4, immunity=True, cure=(0.9, 0.3), **kwargs):
    """ Two strains in one TB-like course: superinfection of the latent, cure for each strain, resistance emerging at failure """
    S, L, I, Tx, R = ss.states('S L I Tx R')
    strains = ss.Strains(['DS', 'RIF'], max_strains=2, split=[0.8, 0.2], rel_trans=[1, 0.6], cure=list(cure),
                         immunity=ss.immunity(due_to=L, rel_sus=0.7) if immunity else None)
    DS, RIF = strains
    return ss.Disease(infection=S >> L @ ss.peryear(beta)*I, activation=L >> I @ ss.peryear(0.3), diagnosis=I >> Tx @ ss.peryear(2),
                      treatment=ss.Transition(Tx, [R, I], p='cure', rate=ss.peryear(2), per='strain'),
                      failure=ss.Transition(Tx, I, rate=ss.peryear(0.2), move=DS >> (DS, RIF)), relapse=R >> S @ ss.years(2),
                      name='tb', strains=strains, **kwargs)


@sc.timer()
def test_strain_clearance():
    sc.heading('Testing strains with courses of their own: clearing one leaves the other\'s stage and clock alone')
    S, I, C = ss.states('S I C')
    hpv = ss.Disease(S, I, C, infection=S >> I @ ss.peryear(3)*I, progression=I >> C @ ss.years(2), clearance=I >> S @ ss.years(1), regression=C >> S @ ss.years(3),
                     init_prev=0.3, name='hpv', strains=ss.Strains(['hpv16', 'hpv18'], shared=False))
    sim = ss.Sim(hpv, n_agents=2000, start=2000, stop=2004, dt=0.25, **kw).init()
    sim.run(until=2002)
    d16, d18 = sim.diseases.hpv16, sim.diseases.hpv18
    both = ss.uids(((d16.I.raw | d16.C.raw) & d18.I.raw).nonzero()[0])
    before = d16.current_state()[both].copy(), d16.ti_I.raw[both].copy(), d16.ti_exit_I.raw[both].copy()
    ss.move(d18.I, d18.S, uids=both)
    assert len(both) and not d18.I.raw[both].any()
    assert all(np.array_equal(a, b, equal_nan=True) for a,b in zip(before, (d16.current_state()[both], d16.ti_I.raw[both], d16.ti_exit_I.raw[both])))
    return sim


@sc.timer()
def test_strain_cure():
    sc.heading('Testing an outcome for each strain carried: a regimen that cures DS only')
    sim = ss.Sim(make_resistance(beta=0, cure=(1.0, 0.0), init=dict(Tx=0.5)), n_agents=4000, start=2000, stop=2003, dt=1/52, **kw).init()
    tb = sim.diseases.tb
    tx = tb.Tx.uids
    sim.people.strain[tx] = np.arange(len(tx)) % 3 # DS, RIF, and both, in turn
    level = np.asarray(sim.people.strain[tx]).copy()
    sim.run(until=2001)
    now = np.asarray(sim.people.strain[tx])
    cured = tb.R.raw[tx] | tb.S.raw[tx]
    assert cured[level == 0].mean() > 0.5 and not cured[level != 0].any() # DS alone is cured; RIF never is
    assert set(now[level == 2]) <= {1, 2} and (now[level == 2] == 1).sum() > 0.5*(level == 2).sum() # Of DS and RIF, RIF is left
    return sim


@sc.timer()
def test_commuting():
    sc.heading('Testing commuters who leave home and come back at finite rates, as a dimension of where people are now')
    beta, gamma, away, N = 1.0, 0.1, 0.1, np.array([8000, 2000.])
    l = 0.05 # Leaving home, and coming back, at rates that give 10% of time away
    r = l*(1 - away)/away
    T = np.array([[1 - away, away], [away, 1 - away]]) # People start in each place for their share of time there
    def make(solver='ode', **kwargs):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(beta) * I), recovery=ss.Transition(I, R, rate=ss.perday(gamma)), init_n=dict(I=ss.by(home=[5, 0])), name='sir')
        sir.stratify(home=['a', 'b']) # Where people live, which does not keep them apart
        sir.stratify(patch=['a', 'b'], kind='space', split=ss.by(N[:, None]/N.sum()*T, dims=['home', 'patch']), # Where they are now, where they mix
                     out=ss.transfer(patch=('a', 'b'), rate=ss.by(home=[ss.perday(l), ss.perday(r)])),
                     back=ss.transfer(patch=('b', 'a'), rate=ss.by(home=[ss.perday(r), ss.perday(l)])))
        return ss.Sim(sir, unit='day', stop=100, total_pop=N.sum(), solver=solver, **kwargs, **kw)

    # Keeling and Rohani's commuter model (7.3): X[i,j] are the residents of i now in j, who leave home at rate l and come back at rate r
    def deriv(t, y):
        S, I, R = y.reshape(3, 2, 2)
        lam = beta*I.sum(axis=0)/y.reshape(3, 2, 2).sum(axis=(0, 1)) # The force of infection where people are now
        move = lambda X: np.array([[r*X[0, 1] - l*X[0, 0], l*X[0, 0] - r*X[0, 1]], [l*X[1, 1] - r*X[1, 0], r*X[1, 0] - l*X[1, 1]]])
        return np.concatenate([-lam*S + move(S), lam*S - gamma*I + move(I), gamma*I + move(R)]).ravel()
    I0 = np.array([5, 0])[:, None]*T
    y = sp_integrate.solve_ivp(deriv, [0, 100], np.concatenate([N[:, None]*T - I0, I0, 0*T]).ravel(), t_eval=np.arange(101), rtol=1e-10, atol=1e-8).y
    sim = make().run()
    I = sim.results.sir.I.by('home').values # By home, wherever people are
    assert np.allclose(I, y.reshape(3, 2, 2, -1)[1].sum(axis=1).T, rtol=1e-4, atol=0.1)

    # As agents, the share of time away is that of the steps: the two transfers are taken one after the other in each daily step, so someone who
    # leaves can come back in the same step, which with a return rate of 0.45 a day gives 7.9% away for the residents of a and 12.4% for those of b
    # (the stationary shares of the daily chain, rather than the ODE's 10%), and the epidemic reaches b
    sim = make(solver='abm', n_agents=10_000).run()
    home, now = sim.people.home.raw, sim.people.patch.raw
    p_l, p_r = -np.expm1(-l), -np.expm1(-r) # The chances of leaving and of coming back in a step
    away_a = p_l*(1 - p_r)/(1 - (1 - p_l)*(1 - p_r)) # Leaving (the transfer out), then perhaps back
    away_b = p_l/(p_r + p_l - p_l*p_r) # Coming back (out), then perhaps leaving again
    assert abs((now[home == 0] == 1).mean() - away_a) < 0.012 and abs((now[home == 1] == 0).mean() - away_b) < 0.025 # About 4 standard deviations
    assert sim.results.sir.infection.cum.by('home').values[-1, 1] > 1000
    with pytest.raises(TypeError, match='one movement'): # Several on one dimension (it was a bare TypeError)
        ss.SIR().stratify(patch=['a', 'b'], mobility=[ss.geo.Mobility(matrix=T), ss.geo.Mobility(matrix=T)])
    return sim


@sc.timer()
def test_register():
    sc.heading('Testing a movement register: people moved, added, and removed on given dates')
    register = sc.dataframe({
        'date':   [0, 10, 20, 30, 40],
        'from':   ['a', None, 'b', 'a', 'c'],
        'to':     ['b', 'c', None, 'c', 'a'],
        'n':      [1000, 500, 300, 200, 100],
        'states': [None, 'S', None, 'R', 'I'],
    })
    def make(data=register, init=ss.by(place=[20, 0, 0])):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3) * I), recovery=ss.Transition(I, R, dur=ss.days(10)), init_n=dict(I=init), name='sir')
        return sir.stratify(place=['a', 'b', 'c'], kind='space', split=[0.5, 0.3, 0.2], mobility=ss.geo.Mobility(data=data))

    sims = sc.objdict()
    for solver in ['ode', 'gillespie', 'abm']:
        with pytest.warns(RuntimeWarning, match='more people than there were'): # Nobody in c is ever infected, so the last event finds nobody to move
            sims[solver] = sim = run(make(), solver, total_pop=10_000, unit='day', stop=60)
        res = sim.results.sir
        n = np.sum([res[s].by('place').values for s in 'SIR'], axis=0) # People in each place, on each day
        assert np.allclose(n[10] - n[9], [0, 0, 500]) # Added to c
        assert np.allclose(n[20] - n[19], [0, -300, 0]) # Removed from b
        assert np.allclose(n[30] - n[29], [-200, 0, 200]) # Moved from a to c
        assert np.allclose(n[40] - n[39], 0) # Nobody to move
        assert np.allclose(n.sum(axis=1)[[0, 15, 25, 60]], [10_000, 10_500, 10_200, 10_200])
        assert res.S.by('place').values[10, 2] - res.S.by('place').values[9, 2] == 500 # Added as susceptible
        assert res.R.by('place').values[30, 2] - res.R.by('place').values[29, 2] == 200 # Only people who had recovered were moved
        if solver != 'abm': # Agents are given their places at random, so the numbers in each start near the split, not on it
            assert np.allclose(n[0], [4000, 4000, 2000]) # An event on the first day happens before it is counted
    assert sims.ode.results.sir.infection.cum.by('place').values[-1, 1] > 1000 # People moved from a took the epidemic to b

    # Stepping gives what the whole run in one call does; a date is placed on the sim's calendar; a whole number is a count of people
    with pytest.warns(RuntimeWarning):
        stepped = ss.Sim(make(), solver='ode', total_pop=10_000, unit='day', stop=60, **kw).run(profile=True)
    assert np.allclose(stepped.results.sir.I.values, sims.ode.results.sir.I.values)
    dated = register.copy()
    dated['date'] = [ss.date('2020-01-01') + ss.days(int(d)) for d in register['date']]
    with pytest.warns(RuntimeWarning):
        sim = ss.Sim(make(dated), solver='ode', total_pop=10_000, unit='day', start='2020-01-01', stop='2020-03-01', **kw).run()
    assert np.allclose(sim.results.sir.S.values, sims.ode.results.sir.S.values)
    res = at_start(ss.Sim(make(register.iloc[:1], init=ss.by(place=[1, 0, 0])), solver='ode', total_pop=10_000, unit='day', stop=5, **kw)).sir
    assert np.allclose(res.I.by('place').values[0], [0.8, 0.2, 0]) # One person in a (not all of them), of whom 1000 in 5000 moved to b

    # With another disease tracked jointly, in either order, everyone is moved once, by the states of the disease that has them (before, the events at the start were lost, and the first disease's states were used: 6% and 44% off)
    def crossed(other_first):
        sir = make()
        S, I = ss.states('S I')
        other = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.2) * I), recovery=ss.Transition(I, S, dur=ss.days(7)), init_prev=0.02, name='other')
        same = ss.modify(sir.rel_sus, multiply=1.0, where=other.I, name='same') # Changes nothing, but the diseases are tracked jointly
        return ss.Sim(diseases=[other, sir] if other_first else [sir, other], connectors=same, solver='ode', total_pop=10_000, unit='day', stop=60, **kw)
    for other_first in [False, True]:
        with pytest.warns(RuntimeWarning):
            res = crossed(other_first).run().results.sir
        assert np.allclose(res.I.values, sims.ode.results.sir.I.values, rtol=1e-4) # To the ODE's accuracy: the other disease's growth can give the joint run more substeps

    # Mistakes
    with pytest.raises(KeyError, match='not a level'):
        ss.Sim(make(register.assign(to=['b', 'd', None, 'c', 'a'])), **kw).run()
    with pytest.raises(KeyError, match='no disease has'):
        ss.Sim(make(register.assign(states=[None, 'X', None, 'R', 'I'])), **kw).run()
    with pytest.raises(KeyError, match='columns'):
        ss.geo.Mobility(data=register.drop(columns='n'))
    with pytest.raises(ValueError, match='migrate'):
        ss.geo.Mobility(data=register, kind='commute')
    # An event can move people to other states in the same place, as SimInf's shift matrix does when animals age
    def ages(data):
        S1, I1 = ss.states('S1 I1')
        S2, I2 = ss.State('S2', people=True), ss.State('I2', people=True)
        dis = ss.Disease(S1, I1, S2, I2, inf1=S1 >> I1 @ 0*I1, inf2=S2 >> I2 @ 0*I2, rec1=I1 >> S1 @ 0.0, rec2=I2 >> S2 @ 0.0, init=dict(I1=0.2), first='S1', name='dis')
        return dis.stratify(herd=['a', 'b'], kind='space', mobility=ss.geo.Mobility(data=data))
    aging = pd.DataFrame({'date':[5, 5], 'from':['a', 'b'], 'to':['a', 'b'], 'n':[100, 50], 'states':['S1, I1']*2, 'to_states':['S2, I2']*2})
    for solver,agents in [('ode', {}), ('gillespie', {}), ('abm', dict(n_agents=1000))]:
        res = ss.Sim(ages(aging), total_pop=1000, stop=10, solver=solver, **agents, **kw).run().results.dis
        assert np.allclose(res.S2.by('herd').values[6] + res.I2.by('herd').values[6], [100, 50])
        assert np.isclose(res.I2.values[6]/(res.S2.values[6] + res.I2.values[6]), 0.2, atol=0.1) # Infected and not move to their own states
    mixed = pd.concat([aging, pd.DataFrame({'date':[6], 'from':['a'], 'to':[None], 'n':[10], 'states':['S1']})]) # Rows with no to_states are read as such
    ss.Sim(ages(mixed), total_pop=1000, stop=10, **kw).run()
    with pytest.raises(ValueError): # As many states to move people to as to move them from
        ss.Sim(ages(aging.assign(states=['S1, I1', 'S1'])), total_pop=1000, stop=10, **kw).init()

    return sims


@sc.timer()
def test_property_values():
    sc.heading('Testing that people hold the values of a numeric dimension, and keep a property they already have')
    # Numeric levels are lower bin edges, and people hold the value (before, they held 0 to 4, binned by the edges 1 to 5, so the top level was empty)
    S, I = ss.states('S I')
    sim = ss.Sim(ss.Disease(inf=S >> I @ ss.peryear(0.0)*I, name='d'), n_agents=5000, start=2000, stop=2001, solver='abm', verbose=0)
    sim.stratify(wealthquintile=[1, 2, 3, 4, 5])
    sim.run()
    assert set(np.unique(sim.people.wealthquintile[sim.people.auids])) == {1, 2, 3, 4, 5}
    res = sim.results.d
    assert np.allclose(res.S.by('wealthquintile').values[0] + res.I.by('wealthquintile').values[0], 1000)

    # A property people already have is kept and binned (before, urban at 0.9 became 0.000), and split= for it raises
    def make(**kw):
        ppl = ss.People(5000, extra_states=[ss.BoolState('urban', default=ss.bernoulli(0.9)), ss.FloatArr('wealth', default=ss.uniform(0, 100))])
        S, I = ss.states('S I')
        sim = ss.Sim(ss.Disease(inf=S >> I @ ss.peryear(0.0)*I, name='d'), people=ppl, start=2000, stop=2001, solver='abm', verbose=0)
        sim.stratify(urban=[False, True], **kw)
        sim.stratify(wealth=[0, 50])
        return sim
    sim = make().run()
    assert np.isclose(sim.people.urban.mean(), 0.9, atol=0.02) and len(np.unique(sim.people.wealth[sim.people.auids])) > 100
    with pytest.raises(ValueError, match='split= would overwrite them'):
        make(split=[0.5, 0.5]).init()
    with pytest.raises(ValueError, match="'age', 'space', 'plain'"): # An unknown kind, e.g. the removed 'clock'
        make(kind='clock')
    return sim


@sc.timer()
def test_many_groups():
    sc.heading('Testing more groups than 16-bit integers hold, as agents')
    S, D = ss.states(2)
    dis = ss.Disease(onset=S >> D @ ss.peryear(1), back=D >> S @ ss.days(7), name='dis')
    dis.stratify(a=[f'a{i}' for i in range(128)], states=[D])
    dis.stratify(b=[f'b{i}' for i in range(384)], states=[D]) # 49,152 groups: an OverflowError, or groups wrapped round to negative numbers
    sim = ss.Sim(diseases=dis, n_agents=2_000, start='2010-01-01', stop='2010-02-01', dt=ss.days(1), verbose=0).run()
    groups = sim.groups
    assert groups.n_groups == 49_152
    assert np.array_equal(groups.group_raw(), groups.level_raw(0).astype(np.int64)*384 + groups.level_raw(1)) # Kept up to date as people move in and out of D
    res = sim.results.dis
    assert res.D.by('b').values[-1].sum() == res.D[-1] == sim.diseases.dis.D.sum()
    return sim


@sc.timer()
def test_given_splits():
    sc.heading('Testing splits given by the people and over several dimensions')
    # Stratifying by a property the people already have keeps it, and is not described as guessed (everyone was put at False, the first level)
    N = 10_000
    tob = np.random.default_rng(1).random(N) < 0.3
    ppl = ss.People(N, extra_states=[ss.BoolState('tob', default=lambda k: tob[:k] if k == N else np.zeros(k, bool))])
    A, B = ss.states('A B')
    d = ss.Disease(A, B, go=A >> B @ ss.peryear(0.1)*ss.by(tob=[0.0, 1.0]), first='A', name='d') # Only smokers move
    sim = ss.Sim(d, people=ppl, stratify=dict(tob=[False, True]), start=2000, stop=2005, dt=1, verbose=0).run()
    assert sim.people.tob.sum() == tob.sum()
    expected = tob.sum()*(1 - np.exp(-0.5))
    assert abs(sim.results.d.go.sum() - expected) < 4*np.sqrt(expected)
    assert 'guessed' not in [line for line in sim.describe(output=True).splitlines() if 'tob' in line and 'dimension' in line][0]

    # A split over two dimensions that contradicts one's own split raises (before, urban came out at 0.497 where 0.17 was given)
    table = np.array([[0.11, 0.21, 0.22, 0.23, 0.23], [0.75, 0.16, 0.05, 0.02, 0.02]]) # P(wealth | urban)
    def make(table):
        S = ss.State()
        m = ss.Disease(stay=ss.Transition(S, S, rate=ss.peryear(0)), name='life')
        m.stratify(urban=[False, True], split=[0.83, 0.17])
        m.stratify(wealth=['w1', 'w2', 'w3', 'w4', 'w5'], split=ss.by(table, dims=['urban', 'wealth']))
        return ss.Sim(m, n_agents=20_000, start=2010, stop=2011, verbose=0)
    with pytest.raises(ValueError, match='0.83'):
        make(table).init()
    n = make(table*np.array([0.83, 0.17])[:, None]).run().results.life.S # As a joint distribution, it agrees
    assert np.isclose(n.select(urban=1).values[0]/n.values[0], 0.17, atol=0.01)
    assert np.isclose(n.select(wealth=0).values[0]/n.values[0], 0.83*0.11 + 0.17*0.75, atol=0.01)
    return sim


def test_sex():
    """ As agents, a dimension of sex is people.female, for everyone including newborns """
    sc.heading('Testing sex as a dimension of agents')
    for kw in [dict(sex=['female', 'male']), dict(sex=['m', 'f']), dict(sex=['female', 'male'], split=[0.7, 0.3])]:
        S, I = ss.states('S I')
        si = ss.Disease(infection=ss.Transition(S, I, rate=0.01 * I), init_prev=0.05, name='si')
        sim = ss.Sim(si, networks=ss.SexualNet(), demographics=ss.Births(birth_rate=30), n_agents=2000, stratify=kw, stop=2010, verbose=0, rand_seed=1).init()
        if 'split' in kw:
            assert np.isclose(sim.people.female.mean(), 0.7, atol=0.03) # people.female follows a split that is given (drawn, for 2000 agents)
        sim.run()
        ppl = sim.people
        strat = sim.groups.strats[0]
        women = ppl[strat.attr].raw[ppl.auids] == strat.female
        assert np.array_equal(women, ppl.female.raw[ppl.auids]), f'{kw}: the levels of sex disagree with people.female'
        assert sim.results.si.S.select(**{strat.name: strat.labels[strat.female]})[-1] == (ppl.female & sim.diseases.si.S).sum()
    with pytest.raises(ValueError, match='people.female'):
        S, I = ss.states('S I')
        ss.Sim(ss.Disease(infection=ss.Transition(S, I, rate=0.01 * I), name='si'), n_agents=100, stratify=dict(sex=['a', 'b']), verbose=0).init()
    return


def test_init_prev_by_agent():
    """ Initial prevalence by a property of agents that is not a stratification: a function of the sim (or of the sim and uids) """
    sc.heading('Testing initial prevalence for each agent')
    def make(init_prev):
        S, I, J = ss.states('S I J')
        return ss.Disease(infection=ss.Transition(S, [I, J], rate=0.0 * I, p=[0.25, 0.75]), recovery=ss.Transition(I, S, dur=ss.years(10)), init_prev=init_prev, name='d')
    women = lambda sim: np.where(sim.people.female, 0.4, 0.0)
    by_uids = lambda sim, uids: np.where(sim.people.age[uids] < 20, 0.5, 0.0)
    for func in [women, by_uids]:
        sim = ss.Sim(make(func), n_agents=20_000, stop=2001, rand_seed=1, **kw).init()
        d, ppl = sim.diseases.d, sim.people
        seeded = d.I | d.J
        if func is women:
            assert not np.any(seeded[ppl.male.uids]) and np.isclose(seeded[ppl.female.uids].mean(), 0.4, atol=0.02)
        else:
            assert not np.any(seeded[(ppl.age >= 20).uids]) and np.isclose(seeded[(ppl.age < 20).uids].mean(), 0.5, atol=0.03)
        assert np.isclose(d.I.count()/seeded.sum(), 0.25, atol=0.03), 'Seeds start wherever a new infection starts'
    with pytest.raises(ss.CapabilityError): # As compartments, only a stratification has a value for each group
        ss.Sim(make(by_uids), solver='ode', **kw).init()
    by_sex = ss.Sim(make(lambda sim: np.where(sim.people.female, 0.4, 0.0)), solver='ode', stratify=dict(sex=['female', 'male']), stop=2001, **kw).init() # A function of a stratification: one value for each group
    n = ss.Sim(make(0.0), solver='ode', stratify=dict(sex=['female', 'male']), stop=2001, **kw).run().results.d.S[0]
    assert np.isclose(by_sex.run().results.d.I.select(sex='female')[0] + by_sex.results.d.J.select(sex='female')[0], 0.4*n/2)
    return sim


if __name__ == '__main__':
    clearance = test_strain_clearance()
    cure    = test_strain_cure()
    commuting = test_commuting()
    register = test_register()
    values  = test_property_values()
    many    = test_many_groups()
    splits  = test_given_splits()
    sex     = test_sex()
    initp   = test_init_prev_by_agent()
