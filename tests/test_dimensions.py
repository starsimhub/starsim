"""
Test dimensions that are more than a property of people: states with dimensions of their own, strains, and places
"""

# %% Imports and settings
import pytest
import numpy as np
import pandas as pd
import scipy.integrate as sp_integrate
import sciris as sc
import starsim as ss
import corpus

kw = dict(verbose=0)


# %% Define the tests


@sc.timer()
def test_strain_cure():
    sc.heading('Testing an outcome for each strain carried: a regimen that cures DS only')
    sim = ss.Sim(corpus.make_resistance(beta=0, cure=(1.0, 0.0), init=dict(Tx=0.5)), n_agents=4000, start=2000, stop=2003, dt=1/52, **kw).init()
    tb = sim.diseases.tb
    tx = tb.Tx.uids
    sim.people.strain[tx] = np.arange(len(tx)) % 3 # DS, RIF, and both, in turn
    level = np.asarray(sim.people.strain[tx]).copy()
    sim.run(until=2001)
    now = np.asarray(sim.people.strain[tx])
    cured = tb.R.raw[tx] | tb.S.raw[tx]
    assert cured[level == 0].mean() > 0.5 and not cured[level != 0].any() # DS alone is cured; RIF never is
    assert set(now[level == 2]) <= {1, 2} and (now[level == 2] == 1).sum() > 0.5*(level == 2).sum() # Of DS and RIF, RIF is left
    assert tb.results.failure.sum() > 0 and tb.results.infection.sum() == 0 # Emergence at treatment failure is not an infection
    return sim


@sc.timer()
def test_strain_levels():
    sc.heading('Testing the strains of a declared subclass in its __init__(), and no strain level for the uninfected')
    class TwoStrains(ss.Disease):
        def __init__(self, **kwargs):
            super().__init__(strains=ss.Strains(['DS', 'RIF'], name='resistance', fractions=[0.5, 0.5]), init_prev=0.2, **kwargs)
            assert self.strains.names == ['DS', 'RIF'] # Before the disease is built
            self.define_states(ss.states('S I R'))
            self.define_transitions(infection=self.S >> self.I @ ss.peryear(2)*self.I, recovery=self.I >> self.R @ ss.years(0.5), waning=self.R >> self.S @ ss.years(0.5))
    sim = ss.Sim(TwoStrains(name='tb'), networks='random', n_agents=1000, start=2000, stop=2002, rand_seed=1, **kw).run()
    tb = sim.diseases.tb
    level = sim.groups.level('resistance')
    assert len(tb.I.uids) and (level[tb.I.uids] >= 0).all()
    assert len(tb.S.uids) and (level[tb.S.uids] == -1).all() and (level[tb.R.uids] == -1).all() # Not DS, the first strain
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
        sir.stratify(patch=ss.levels(['a', 'b'], kind='space', fractions=ss.by(N[:, None]/N.sum()*T, dims=['home', 'patch'])), # Where they are now, where they mix
                     out=ss.transfer(dict(patch=('a', 'b')), rate=ss.by(home=[ss.perday(l), ss.perday(r)])),
                     back=ss.transfer(dict(patch=('b', 'a')), rate=ss.by(home=[ss.perday(r), ss.perday(l)])))
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
        return sir.stratify(place=ss.levels({'a': 0.5, 'b': 0.3, 'c': 0.2}, kind='space', mobility=ss.geo.register(data)))

    sims = sc.obj()
    for solver in ['ode', 'abm']:
        with pytest.warns(RuntimeWarning, match='more people than there were'): # Nobody in c is ever infected, so the last event finds nobody to move
            sims[solver] = sim = corpus.run_with(make(), solver, total_pop=10_000, unit='day', stop=60)
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

    # A date is placed on the sim's calendar; a whole number is a count of people
    dated = register.copy()
    dated['date'] = [ss.date('2020-01-01') + ss.days(int(d)) for d in register['date']]
    with pytest.warns(RuntimeWarning):
        sim = ss.Sim(make(dated), solver='ode', total_pop=10_000, unit='day', start='2020-01-01', stop='2020-03-01', **kw).run()
    assert np.allclose(sim.results.sir.S.values, sims.ode.results.sir.S.values)
    res = corpus.at_start(ss.Sim(make(register.iloc[:1], init=ss.by(place=[1, 0, 0])), solver='ode', total_pop=10_000, unit='day', stop=5, **kw)).sir
    assert np.allclose(res.I.by('place').values[0], [0.8, 0.2, 0]) # One person in a (not all of them), of whom 1000 in 5000 moved to b

    # An event can move people to other states in the same place, as SimInf's shift matrix does when animals age
    def ages(data):
        S1, I1 = ss.states('S1 I1')
        S2, I2 = ss.State('S2', people=True), ss.State('I2', people=True)
        dis = ss.Disease(S1, I1, S2, I2, inf1=S1 >> I1 @ 0*I1, inf2=S2 >> I2 @ 0*I2, rec1=I1 >> S1 @ 0.0, rec2=I2 >> S2 @ 0.0, init=dict(I1=0.2), default_state='S1', name='dis')
        return dis.stratify(herd=ss.levels(['a', 'b'], kind='space', mobility=ss.geo.register(data)))
    aging = pd.DataFrame({'date':[5, 5], 'from':['a', 'b'], 'to':['a', 'b'], 'n':[100, 50], 'states':['S1, I1']*2, 'to_states':['S2, I2']*2})
    for solver,agents in [('ode', {}), ('abm', dict(n_agents=1000))]:
        res = ss.Sim(ages(aging), total_pop=1000, stop=10, solver=solver, **agents, **kw).run().results.dis
        assert np.allclose(res.S2.by('herd').values[6] + res.I2.by('herd').values[6], [100, 50])
        assert np.isclose(res.I2.values[6]/(res.S2.values[6] + res.I2.values[6]), 0.2, atol=0.1) # Infected and not move to their own states

    return sims


@sc.timer()
def test_property_values():
    sc.heading('Testing that people hold the values of a numeric dimension (lower bin edges)')
    S, I = ss.states('S I')
    sim = ss.Sim(ss.Disease(inf=S >> I @ ss.peryear(0.0)*I, name='d'), n_agents=5000, start=2000, stop=2001, solver='abm', verbose=0)
    sim.stratify(wealthquintile=[1, 2, 3, 4, 5])
    sim.run()
    assert set(np.unique(sim.people.wealthquintile[sim.people.auids])) == {1, 2, 3, 4, 5}
    res = sim.results.d
    assert np.allclose(res.S.by('wealthquintile').values[0] + res.I.by('wealthquintile').values[0], 1000)

    return sim
    sim = make().run()
    assert np.isclose(sim.people.urban.mean(), 0.9, atol=0.02) and len(np.unique(sim.people.wealth[sim.people.auids])) > 100
    with pytest.raises(ValueError, match='would overwrite them'):
        make(urban={False: 0.5, True: 0.5}).init()
    return sim


@sc.timer()
def test_given_splits():
    sc.heading('Testing splits given by the people and over several dimensions')
    # Stratifying by a property the people already have keeps it, and is not described as guessed (everyone was put at False, the first level)
    N = 10_000
    tob = np.random.default_rng(1).random(N) < 0.3
    ppl = ss.People(N, extra_states=[ss.BoolState('tob', default=lambda k: tob[:k] if k == N else np.zeros(k, bool))])
    A, B = ss.states('A B')
    d = ss.Disease(A, B, go=A >> B @ ss.peryear(0.1)*ss.by(tob=[0.0, 1.0]), default_state='A', name='d') # Only smokers move
    sim = ss.Sim(d, people=ppl, stratify=dict(tob=[False, True]), start=2000, stop=2005, dt=1, verbose=0).run()
    assert sim.people.tob.sum() == tob.sum()
    expected = tob.sum()*(1 - np.exp(-0.6)) # Six steps, from 2000 to 2005
    assert abs(sim.results.d.go.sum() - expected) < 4*np.sqrt(expected)
    assert 'Guessed' not in [line for line in sim.describe().splitlines() if 'tob' in line and 'Dimension' in line][0]

    # The people's ages are kept, so a split of the age groups must agree with them
    def aged(age_data):
        S = ss.State()
        m = ss.Disease(stay=ss.Transition(S, S, rate=ss.peryear(0)), name='life')
        m.stratify(age={0: 0.2, 18: 0.5, 65: 0.3})
        return ss.Sim(m, people=ss.People(5_000, age_data=age_data), stop=1, verbose=0)
    with pytest.raises(ValueError, match='agents keep their ages'):
        aged(np.array([[0, 1], [18, 0]])).init()
    ages = aged(np.array([[0, 0.2], [18, 0.5], [65, 0.3], [90, 0]])).init().people.age
    assert np.isclose((ages >= 65).mean(), 0.3, atol=0.02)

    # A split over two dimensions that contradicts one's own split raises
    table = np.array([[0.11, 0.21, 0.22, 0.23, 0.23], [0.75, 0.16, 0.05, 0.02, 0.02]]) # P(wealth | urban)
    def make(table):
        S = ss.State()
        m = ss.Disease(stay=ss.Transition(S, S, rate=ss.peryear(0)), name='life')
        m.stratify(urban={False: 0.83, True: 0.17})
        m.stratify(wealth=ss.levels(['w1', 'w2', 'w3', 'w4', 'w5'], fractions=ss.by(table, dims=['urban', 'wealth'])))
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
    for kw in [dict(sex=['m', 'f']), dict(sex={'female': 0.7, 'male': 0.3})]:
        S, I = ss.states('S I')
        si = ss.Disease(infection=ss.Transition(S, I, rate=0.01 * I), init_prev=0.05, name='si')
        sim = ss.Sim(si, networks=ss.SexualNet(), demographics=ss.Births(birth_rate=0.03), n_agents=2000, stratify=kw, stop=2010, verbose=0, rand_seed=1).init()
        if isinstance(kw['sex'], dict):
            assert np.isclose(sim.people.female.mean(), 0.7, atol=0.03) # people.female follows a split that is given (drawn, for 2000 agents)
        start = (sim.people.female & sim.diseases.si.S).sum()
        sim.run()
        ppl = sim.people
        strat = sim.groups.strats[0]
        women = ppl[strat.attr].raw[ppl.auids] == strat.female
        assert np.array_equal(women, ppl.female.raw[ppl.auids]), f'{kw}: the levels of sex disagree with people.female'
        assert sim.results.si.S.select(**{strat.name: strat.labels[strat.female]})[0] == start # The women in S at the start
    return


def test_init_prev_by_agent():
    """ Initial prevalence by a property of agents that is not a stratification: a function of the sim (or of the sim and uids) """
    sc.heading('Testing initial prevalence for each agent')
    def make(init_prev):
        S, I, J = ss.states('S I J')
        return ss.Disease(infection=ss.Transition(S, {I: 0.25, J: 0.75}, rate=0.0 * I), recovery=ss.Transition(I, S, dur=ss.years(10)), init_prev=init_prev, name='d')
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


@sc.timer()
def test_places_by_name():
    sc.heading('Testing starting values by place name, and the note for names that are not places')
    def make(init_n=None, init_prev=None, beta=0.3):
        S, I, R = ss.states('S I R')
        return ss.Disease(S >> I @ beta*I, I >> R @ ss.days(8), init_n=init_n, init_prev=init_prev, name='m')
    patch = dict(patch=ss.levels(['a', 'b', 'c'], kind='space'))

    # A place left out of a starting value by name starts at 0, as agents and as compartments, but a rate needs every place
    for n_agents in [None, 2000]:
        given = dict(stratify=patch, stop=3, unit='day', **kw) | (dict(n_agents=n_agents) if n_agents else {})
        by_name = ss.Sim(make(init_n=dict(I=ss.by(patch={'b': 10}))), **given).run().results.m.I.by('patch').values[0]
        in_order = ss.Sim(make(init_n=dict(I=ss.by(patch=[0, 10, 0]))), **given).run().results.m.I.by('patch').values[0]
        assert np.allclose(by_name, in_order) and in_order[1] > 0
    seed = ss.by(patch={'c': 0.1})
    prev = ss.Sim(make(init_prev=seed), stratify=patch, stop=3, unit='day', **kw).run().results.m.I.by('patch').values[0]
    assert prev[0] == prev[1] == 0 and prev[2] > 0
    assert seed.rest is None # A copy is the starting value, so the same ss.by elsewhere still needs every place
    with pytest.raises(ValueError): # A misspelled place is not quietly 0
        ss.Sim(make(init_n=dict(I=ss.by(patch={'bb': 10}))), stratify=patch, stop=3, unit='day', **kw).run()

    # A name like patch given only names mixes as one population, which a note says; kind= given says what it is, as do contacts by it
    notes = lambda nets=None, **strat: [note for note in ss.Sim(make(init_prev=0.01), stratify=strat, stop=3, unit='day', **kw, **(dict(networks=nets) if nets else {})).init().notes if 'mix as one population' in note]
    assert not notes(ss.contacts(region=[[0.8, 0.2], [0.2, 0.8]]), region=['north', 'south'])
    assert len(notes(region=['north', 'south'])) == 1
    assert not notes(region=ss.levels(['north', 'south'], kind='plain')) and not notes(region=ss.levels(['north', 'south'], kind='space')) and not notes(risk=['low', 'high'])
    return


@sc.timer()
def test_levels():
    sc.heading('Testing ss.levels(): options with the levels, several dimensions in one call, and who= as a condition')
    def make():
        S, I, R = ss.states('S I R')
        return ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(8), init_prev=0.01, name='sir')
    for solver in ['ode', 'abm']: # Parity only for women of 15 and over, and severity only for the infected, in one call
        sir = make().stratify(sex=['female', 'male'], age=[0, 15, 50], parity=ss.levels({'0': 0.4, '1+': 0.6}, who=ss.person.female & (ss.person.age >= 15)), severity=ss.levels({'mild': 0.9, 'severe': 0.1}, states=['I']))
        sim = ss.Sim(sir, solver=solver, n_agents=2000, total_pop=2000, unit='day', stop=10, verbose=0).run()
        g = sim.groups
        d = [strat.name for strat in g.strats].index('parity')
        assert np.array_equal(g.group_inside[:, d], (g.group_coords[:, 0] == 0) & (g.group_coords[:, 1] >= 1)), f'{solver}: parity only for women of 15 and over'
    assert [s.name for s in make().stratify(states=['low', 'high'], kind=[0, 1]).graph.strats] == ['states', 'kind'] # An option's name can be a dimension's

    # ss.levels(fractions=) as its docstring gives it: the shares at each combination of levels
    sim = ss.Sim(diseases=ss.SIR(beta=0.1), stop=2002, verbose=0)
    sim.stratify(age={0: 0.5, 15: 0.5}, risk=ss.levels(['low', 'high'], fractions=ss.by(np.array([[0.45, 0.05], [0.2, 0.3]]), dims=['age', 'risk'])))
    sim.run()
    return sim


def test_levels_effects():
    sc.heading('ss.levels(rel_sus=, rel_trans=): a dimension with its own effect, the same as a modifier by level')
    def make(effect):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(S >> I @ 0.2*I, I >> R @ ss.days(10), init_prev=0.01, name='sir')
        sir.stratify(vaccinated=ss.levels({False: 0.3, True: 0.7}, states=['S'], **(dict(rel_sus=[1, 0.1]) if effect else {})), risk=ss.levels({'lo': 0.7, 'hi': 0.3}, **(dict(rel_trans=[1, 3]) if effect else {})))
        mods = None if effect else [ss.modify('sir.rel_sus', multiply=ss.by(vaccinated=[1, 0.1])), ss.modify('sir.rel_trans', multiply=ss.by(risk=[1, 3]))]
        return sir, mods
    for solver in ['ode', 'abm']:
        size = dict(total_pop=5000) if solver == 'ode' else dict(n_agents=5000)
        totals = [ss.Sim(sir, connectors=mods, stop=ss.days(100), solver=solver, **size, **kw).run().results.sir.infection.sum() for sir,mods in [make(True), make(False)]]
        assert totals[0] == totals[1], f'ss.levels(rel_sus=, rel_trans=) should be the same as the modifiers ({solver})'
    sir, _ = make(True)
    sim = ss.Sim(sir, total_pop=1000, **kw)
    assert 'rel_sus: False 1, True 0.1' in str(sim.describe()) and 'by level of vaccinated' in str(sim.what_modifies(sir.rel_sus))
    return


def test_levels_R0():
    sc.heading('Protection as a level of S lowers R0 itself, while immunity as starting in R lowers only the R at the start (as in T7)')
    def make(**kwargs):
        S, I, R = ss.states('S I R')
        return ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), name='d', **kwargs)
    base = make().R0()
    leaky = make(init_n=dict(I=10))
    leaky.stratify(vaccinated=ss.levels({False: 0.5, True: 0.5}, states=['S'], rel_sus=[1, 0.1]))
    sim = ss.Sim(leaky, total_pop=10_000, unit='day', stop=10, **kw).run()
    R0, R = sim.diseases.d.R0(), sim.diseases.d.Rt().values[0]
    assert np.isclose(R0, 0.55*base, rtol=0.01) and np.isclose(R, R0, rtol=0.01) # Each person in S counts with their susceptibility
    immune = ss.Sim(make(init=dict(R=0.5), init_n=dict(I=10)), total_pop=10_000, unit='day', stop=10, **kw).run()
    R0, R = immune.diseases.d.R0(), immune.diseases.d.Rt().values[0]
    assert np.isclose(R0, base, rtol=0.01) and np.isclose(R, 0.5*base, rtol=0.01)
    return


if __name__ == '__main__':
    for name,test in list(globals().items()):
        if name.startswith('test_'):
            test()
