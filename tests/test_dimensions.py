"""
Test dimensions that are more than a property of people: states with dimensions of their own, strains, and places
"""

# %% Imports and settings
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

kw = dict(verbose=0)
solvers = ['ode', 'binomial', 'abm']


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
    Sv, Ev, Iv     = ss.states(3)
    Nh             = (Sh + Eh + Ih + Rh).n
    malaria = ss.Disease(
        infection  = Sh >> Eh @ a * b * Iv.n / Nh,
        latency    = Eh >> Ih @ ss.days(12),
        recovery   = Ih >> Rh @ ss.days(100),
        waning     = Rh >> Sh @ ss.days(365),
        emergence  = ss.null >> Sv @ ss.perday(5_000),
        biting     = Sv >> Ev @ a * c * Ih.n / Nh,
        incubation = Ev >> Iv @ ss.days(10),
        mortality  = (Sv | Ev | Iv) >> ss.null @ ss.days(10),
        init       = dict(Ih=0.01, Sv=50_000),
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
    )
    tb.stratify(
        resistance = ['DS', 'MDR'],
        split      = [1.0, 0.0],
        adjust     = {'infection': [1.0, 0.7], 'cure': [1.0, 0.3]},
        acquired   = ss.transfer(resistance='DS -> MDR', rate=ss.peryear(0.02), states=['on_treatment']),
    )
    return tb


def make_towns(**kwargs):
    """ An SIR model in three towns, with nobody infected """
    towns = ss.geo(names=['north', 'south', 'east'], pop=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
    sir = ss.Disease(
        infection = ss.transmission('S -> I', beta=0.3),
        recovery  = ss.progression('I -> R', dur=10),
        name = 'sir',
    )
    return sir.stratify(patch=towns, **kwargs)


def seed_north(sim):
    """ Infect 50 people in the first town """
    if sim.ti == 0:
        sir = sim.diseases.sir
        if sim.solver == 'abm':
            uids = (sim.people.patch == 0).uids[:5] # Each agent stands for 10 people
            sir.S[uids] = False
            sir.I[uids] = True
        else:
            engine = sir.engine
            engine.y[engine.state_comps['S'][0], 0] -= 50
            engine.y[engine.state_comps['I'][0], 0] += 50
    return


# %% Define the tests

@sc.timer()
def test_states():
    sc.heading('Testing dimensions that apply to some states only')
    sims = sc.objdict()
    for solver in solvers:
        malaria = make_malaria()
        states = malaria.graph.states
        malaria.stratify(species=['gambiae', 'funestus'], split=[0.75, 0.25], states=[states.Sv, states.Ev, states.Iv], adjust={'mortality': [1.0, 2.0]})
        sims[solver] = run(malaria, solver, unit='day', stop=2*365, total_pop=20_000, stratify=dict(age=[0, 5, 15]))
    res = sims.ode.results.malaria
    assert [s.name for s in malaria.graph.count_states] == ['Sv', 'Ev', 'Iv'] # Nobody moves between humans and mosquitoes, so the mosquitoes are counts
    assert np.allclose(res.n_Sv.by('species').values[0], [37_500, 12_500]) # Split by species ...
    assert np.allclose(res.emergence.by('species').values[5], [3750, 1250])
    assert np.allclose(sum(res[f'n_{s}h'] for s in 'SEIR'), 20_000) # ... and left out of the population
    n_Iv = res.n_Iv.by('species').values[-1]
    assert n_Iv[1] < n_Iv[0]/6, 'Expected fewer of the species that is rarer and dies faster'
    assert res.n_Ih.by('age').values.shape[1] == 3
    with pytest.raises(KeyError): # A state has only the dimensions it was stratified by
        res.n_Iv.by('age')
    with pytest.raises(KeyError):
        res.n_Ih.by('species')
    for solver in ['binomial', 'abm']:
        other = sims[solver].results.malaria
        assert np.allclose(other.n_Iv.by('species').values[-1], n_Iv, rtol=0.2), f'Expected {solver} to match the ODE'
        assert np.isclose(other.n_Ih[-1], res.n_Ih[-1], rtol=0.1)

    # A dimension of the people in some states: they take a level when they enter one, and lose it when they leave
    def make():
        S, I = ss.states(2)
        hiv = ss.Disease(infection=S >> I @ 0.2 * I, recovery=I >> S @ 0.05, init_prev=0.05, name='hiv')
        hiv.stratify(stage=['early', 'mid', 'late'], split=[1, 0, 0], states=[I], adjust={'recovery': [1.0, 0.5, 0.25]},
                     decline=ss.transfer(stage='early -> mid', rate=0.05), decline2=ss.transfer(stage='mid -> late', rate=0.05))
        return hiv
    staged = sc.objdict({solver:run(make(), solver, unit='day', stop=300, total_pop=20_000) for solver in solvers})
    res = staged.ode.results.hiv
    assert np.allclose(res.infection.by('stage').values.sum(axis=0)[1:], 0) # Everyone starts in the first stage
    assert getattr(res.n_S, 'strata', None) is None # Susceptible people have no stage
    for solver in ['binomial', 'abm']:
        other = staged[solver].results.hiv
        assert np.allclose(other.n_I.by('stage').values[-1], res.n_I.by('stage').values[-1], rtol=0.15), f'Expected {solver} to match the ODE'

    with pytest.raises(ss.CodedError, match='E204'):
        make_malaria().stratify(species=['gambiae', 'funestus'], states=['Sv', 'Evv'])

    # A dimension of some states only, entered with a split that depends on age, as summer2's clinical strata
    def clinical(adjust=None):
        seir = ss.Disease(infection=ss.transmission('S -> E', beta=0.4, source='I'), onset=ss.progression('E -> I', dur=3), recovery=ss.progression('I -> R', dur=7), init=dict(I=0.01), name='seir')
        seir.stratify(age=[0, 20, 60], split=[0.3, 0.5, 0.2])
        seir.stratify(clinical=['mild', 'severe'], states=['I'], split=ss.by([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]], dims=['age', 'clinical']), adjust=adjust)
        return seir
    for solver,agents in [('ode', {}), ('binomial', {}), ('abm', dict(n_agents=50_000))]:
        res = ss.Sim(clinical(), solver=solver, stop=100, **agents, **kw).run().results.seir
        severe = [res.n_I.select(age=age, clinical='severe').values.sum()/res.n_I.select(age=age).values.sum() for age in ['0-19', '20-59', '60+']]
        assert np.allclose(severe, [0.1, 0.3, 0.8], atol=0.03), f'Expected the share severe to depend on age with {solver}'
        assert np.allclose([res.n_S.select(age=age).values[0] for age in ['0-19', '20-59', '60+']], np.array([0.3, 0.5, 0.2])*res.n_S.values[0], rtol=0.02)
    # A flow into such a dimension is recorded at the level people arrive at
    for solver,agents in [('ode', {}), ('abm', dict(n_agents=20_000))]:
        onset = ss.Sim(clinical(), solver=solver, stop=100, **agents, **kw).run().results.seir.onset.by('clinical').values.sum(axis=0)
        assert np.isclose(onset[1]/onset.sum(), 0.3*0.1 + 0.5*0.3 + 0.2*0.8, atol=0.02)

    # A starting count by level of such a dimension is that many people, and the first state has everyone else
    seir = clinical()
    seir.graph.states.I.init = ss.by(clinical=[10, 0])
    res = ss.Sim(seir, total_pop=1000, stop=2, **kw).run().results.seir
    assert np.isclose(res.n_I.select(clinical='mild').values[0], 10) and np.isclose(res.n_I.values[0], 10)
    assert np.isclose(res.n_S.values[0], 990)

    with pytest.raises(ss.CodedError, match='E306'): # People in E have no clinical level, so an adjustment of their onset cannot apply
        ss.Sim(clinical(adjust={'onset': [1, 2]}), **kw).init()

    # Infectiousness by clinical level: none at all stops transmission, and the solvers agree on the rest
    assert ss.Sim(clinical(adjust={'rel_trans': [0, 0]}), stop=100, **kw).run().results.seir.n_R.values[-1] < 101
    sizes = [ss.Sim(clinical(adjust={'rel_trans': [0.5, 2]}), solver=solver, stop=150, total_pop=20_000, **agents, **kw).run().results.seir.n_R.values[-1]
             for solver,agents in [('ode', {}), ('abm', dict(n_agents=20_000))]]
    assert np.isclose(*sizes, rtol=0.05)
    with pytest.raises(ss.CodedError, match='E306'): # Susceptible people have no clinical level
        ss.Sim(clinical(adjust={'rel_sus': [1, 2]}), **kw).init()
    return sims


@sc.timer()
def test_strains():
    sc.heading('Testing a strain: a new infection takes the strain of its source')
    sims = sc.objdict()
    for solver in solvers:
        sims[solver] = run(make_tb(), solver, start=2000, stop=2050, dt=ss.years(0.1), total_pop=100_000)
    tb = sims.ode.diseases.tb
    res = tb.results
    assert tb.graph.strats[0].states == ['latent', 'active', 'on_treatment'] # A susceptible person has no strain
    assert getattr(res.n_susceptible, 'strata', None) is None
    by_strain = res.cum_infection.by('resistance').values[-1]
    assert res.n_active.by('resistance').values[0, 1] == 0 # Nobody starts with a resistant strain
    assert res.cum_acquired[-1] > 0 and by_strain[1] > 0 # It is acquired during treatment, and then transmitted
    assert by_strain[1] < 0.1*by_strain[0]
    for solver in ['binomial', 'abm']:
        other = sims[solver].results.tb
        assert np.isclose(other.cum_infection[-1], res.cum_infection[-1], rtol=0.05), f'Expected {solver} to match the ODE'
        assert np.isclose(other.n_active.by('resistance').values[-1, 0], res.n_active.by('resistance').values[-1, 0], rtol=0.1)
        assert 0 < other.cum_infection.by('resistance').values[-1, 1] < 0.1*by_strain[0]

    # Without acquired resistance, the resistant strain never appears; and a fitter strain takes over
    def make(adjust):
        sis = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> S', dur=10), init_prev=0.01, name='sis')
        return sis.stratify(strain=['wild', 'variant'], split=[0.9, 0.1], adjust={'infection': adjust})
    for solver in solvers:
        res = run(make([1.0, 1.5]), solver, stop=300, total_pop=50_000).results.sis
        n_I = res.n_I.by('strain').values
        assert n_I[0, 0] > 5*n_I[0, 1] and n_I[-1, 1] > 5*n_I[-1, 0], f'Expected the more transmissible strain to take over with {solver}'
    # Cross-immunity between strains: complete protection is the base model; none lets each person have each strain (as agents, who keep every strain they have had)
    def flu(immunity):
        dis = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=7), init_prev=0.01, name='flu')
        return dis.stratify(strain=['A', 'B'], immunity=immunity)
    total = lambda immunity, **kwargs: ss.Sim(flu(immunity), unit='day', stop=200, **kwargs, **kw).run().results.flu.infection.sum()
    once = total(None)
    assert np.isclose(total(np.ones((2, 2))), once)
    twice = total(np.eye(2), n_agents=10_000)
    assert 1.5*once < twice < 2.2*once
    assert total(np.eye(2)) > twice # A compartmental run keeps only the last strain, so people can go from A to B and back
    with pytest.raises(ss.CodedError, match='E603'):
        flu(np.eye(3))

    return sims


@sc.timer()
def test_space():
    sc.heading('Testing places, with and without movement between them')
    mobility = sc.objdict(
        none    = None,
        gravity = ss.gravity(k=2e-4), # Commuting, with the share of time away from the gravity model
        away    = ss.Mobility(ss.gravity(), away=0.2), # Commuting, with 20% of time spent away
        visit   = ss.Mobility(ss.gravity(), away=0.2, kind='visit'), # The same, but travelers do not infect the people they visit
        migrate = ss.Mobility(ss.gravity(k=2e-5), kind='migrate'),
        force   = ss.Mobility(ss.gravity(), away=0.2, kind='force'), # Nobody moves; 20% of each town's force of infection goes to the others
    )
    peaks = sc.objdict()
    for key,mob in mobility.items():
        for solver in solvers:
            sim = ss.Sim(make_towns(mobility=mob), stop=150, unit='day', solver=solver, total_pop=100_000, **(dict(n_agents=10_000) if solver == 'abm' else {}), **kw)
            sim.init()
            sim.loop.insert(seed_north, label='sim.start_step')
            sim.run()
            res = sim.results.sir
            attack = res.cum_infection.by('patch').values[-1]/[50_000, 30_000, 20_000]
            if key == 'none': # People mix within their own town, so the epidemic stays where it started
                assert attack[0] > 0.9 and np.allclose(attack[1:], 0)
            else:
                assert np.all(attack > 0.85), f'Expected the epidemic to reach every town with {key} and {solver}, not {attack}'
            peaks[f'{key}_{solver}'] = res.n_I.by('patch').values.argmax(axis=0)
    assert peaks.gravity_ode[0] < peaks.gravity_ode[1] < peaks.gravity_ode[2] # The nearer town is reached first
    assert np.ptp(peaks.away_ode) < 3 < np.ptp(peaks.gravity_ode) # The more time people spend away, the closer together the towns peak
    text = sim.describe(output=True)
    assert 'patch (space): north, south, east' in text and "kind='force'" in text

    # A moving force of infection goes only where the matrix sends it: from a to b, not from b to a
    for matrix,reached in [([[0, 0.1], [0, 0]], True), ([[0, 0], [0.1, 0]], False)]:
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init=dict(I=ss.by(place=[10, 0])), name='sir')
        sir.stratify(place=['a', 'b'], kind='space', mobility=ss.Mobility(matrix=matrix, kind='force'))
        for solver in ['ode', 'abm']:
            attack = run(sir, solver, total_pop=20_000, unit='day', stop=200).results.sir.cum_infection.by('place').values[-1]
            assert attack[0] > 5000 and (attack[1] > 5000) == reached, f'Expected b to be reached: {reached}, with {solver}, not {attack}'

    # A count has one value per place, and a count in a rate is the count in the same place
    S, I, R, W = ss.states(4)
    cholera = ss.Disease(
        infection = S >> I @ 0.5 * W.n / (W.n + 1e5),
        recovery  = I >> R @ ss.days(5),
        shedding  = ss.null >> W @ 10 * I.n,
        decay     = W >> ss.null @ ss.days(30),
        init      = dict(I=0.001),
        name      = 'cholera',
    )
    cholera.stratify(patch=['upstream', 'downstream'], adjust={'shedding': [1.0, 0.0]})
    for solver in ['ode', 'abm']:
        res = run(cholera, solver, total_pop=100_000, unit='day', stop=200).results.cholera
        assert res.n_W.by('patch').values[:, 1].max() == 0 # Nobody sheds downstream ...
        assert res.cum_infection.by('patch').values[-1, 1] == 0 # ... so nobody is infected there
        assert res.cum_infection.by('patch').values[-1, 0] > 40_000

    # A gravity model needs a geography
    with pytest.raises(ss.CodedError, match='E402'):
        sir = ss.Disease(recovery=ss.progression('I -> R', dur=3)).stratify(patch=['a', 'b'], mobility=ss.gravity())
        ss.Sim(sir, **kw).init()
    return peaks


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
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.3)), recovery=ss.progression('I -> R', dur=ss.days(10)), init=dict(I=init), name='sir')
        return sir.stratify(place=['a', 'b', 'c'], kind='space', split=[0.5, 0.3, 0.2], mobility=ss.Mobility(data=data))

    sims = sc.objdict()
    for solver in ['ode', 'gillespie', 'abm']:
        with pytest.warns(RuntimeWarning, match='more people than there were'): # Nobody in c is ever infected, so the last event finds nobody to move
            sims[solver] = sim = run(make(), solver, total_pop=10_000, unit='day', stop=60)
        res = sim.results.sir
        n = np.sum([res[f'n_{s}'].by('place').values for s in 'SIR'], axis=0) # People in each place, on each day
        assert np.allclose(n[10] - n[9], [0, 0, 500]) # Added to c
        assert np.allclose(n[20] - n[19], [0, -300, 0]) # Removed from b
        assert np.allclose(n[30] - n[29], [-200, 0, 200]) # Moved from a to c
        assert np.allclose(n[40] - n[39], 0) # Nobody to move
        assert np.allclose(n.sum(axis=1)[[0, 15, 25, 60]], [10_000, 10_500, 10_200, 10_200])
        assert res.n_S.by('place').values[10, 2] - res.n_S.by('place').values[9, 2] == 500 # Added as susceptible
        assert res.n_R.by('place').values[30, 2] - res.n_R.by('place').values[29, 2] == 200 # Only people who had recovered were moved
        if solver != 'abm': # Agents are given their places at random, so the numbers in each start near the split, not on it
            assert np.allclose(n[0], [4000, 4000, 2000]) # An event on the first day happens before it is counted
    assert sims.ode.results.sir.cum_infection.by('place').values[-1, 1] > 1000 # People moved from a took the epidemic to b

    # Stepping gives what the whole run in one call does; a date is placed on the sim's calendar; a whole number is a count of people
    with pytest.warns(RuntimeWarning):
        stepped = ss.Sim(make(), solver='ode', total_pop=10_000, unit='day', stop=60, **kw).run(profile=True)
    assert np.allclose(stepped.results.sir.n_I.values, sims.ode.results.sir.n_I.values)
    dated = register.copy()
    dated['date'] = [ss.date('2020-01-01') + ss.days(int(d)) for d in register['date']]
    with pytest.warns(RuntimeWarning):
        sim = ss.Sim(make(dated), solver='ode', total_pop=10_000, unit='day', start='2020-01-01', stop='2020-03-01', **kw).run()
    assert np.allclose(sim.results.sir.n_S.values, sims.ode.results.sir.n_S.values)
    res = ss.Sim(make(register.iloc[:1], init=ss.by(place=[1, 0, 0])), solver='ode', total_pop=10_000, unit='day', stop=5, **kw).run().results.sir
    assert np.allclose(res.n_I.by('place').values[0], [0.8, 0.2, 0]) # One person in a (not all of them), of whom 1000 in 5000 moved to b

    # Mistakes
    with pytest.raises(KeyError, match='not a level'):
        ss.Sim(make(register.assign(to=['b', 'd', None, 'c', 'a'])), **kw).run()
    with pytest.raises(KeyError, match='no disease has'):
        ss.Sim(make(register.assign(states=[None, 'X', None, 'R', 'I'])), **kw).run()
    with pytest.raises(KeyError, match='columns'):
        ss.Mobility(data=register.drop(columns='n'))
    with pytest.raises(ValueError, match='migrate'):
        ss.Mobility(data=register, kind='commute')
    # An event can move people to other states in the same place, as SimInf's shift matrix does when animals age
    def ages(data):
        S1, I1 = ss.states('S1 I1')
        S2, I2 = ss.State('S2', people=True), ss.State('I2', people=True)
        dis = ss.Disease(S1, I1, S2, I2, inf1=S1 >> I1 @ 0*I1, inf2=S2 >> I2 @ 0*I2, rec1=I1 >> S1 @ 0.0, rec2=I2 >> S2 @ 0.0, init=dict(I1=0.2), name='dis')
        return dis.stratify(herd=['a', 'b'], kind='space', mobility=ss.Mobility(data=data))
    aging = pd.DataFrame({'date':[5, 5], 'from':['a', 'b'], 'to':['a', 'b'], 'n':[100, 50], 'states':['S1, I1']*2, 'to_states':['S2, I2']*2})
    for solver,agents in [('ode', {}), ('gillespie', {}), ('abm', dict(n_agents=1000))]:
        res = ss.Sim(ages(aging), total_pop=1000, stop=10, solver=solver, **agents, **kw).run().results.dis
        assert np.allclose(res.n_S2.by('herd').values[6] + res.n_I2.by('herd').values[6], [100, 50])
        assert np.isclose(res.n_I2.values[6]/(res.n_S2.values[6] + res.n_I2.values[6]), 0.2, atol=0.1) # Infected and not move to their own states
    mixed = pd.concat([aging, pd.DataFrame({'date':[6], 'from':['a'], 'to':[None], 'n':[10], 'states':['S1']})]) # Rows with no to_states are read as such
    ss.Sim(ages(mixed), total_pop=1000, stop=10, **kw).run()
    with pytest.raises(ValueError): # As many states to move people to as to move them from
        ss.Sim(ages(aging.assign(states=['S1, I1', 'S1'])), total_pop=1000, stop=10, **kw).init()

    return sims


@sc.timer()
def test_mixing():
    sc.heading('Testing contact matrices by dimension, and where people start')
    def make():
        sir = ss.Disease(
            infection = ss.transmission('S -> I', beta=2),
            recovery  = ss.progression('I -> R', rate=1/3),
            init      = dict(I=10),
            name      = 'sir',
        )
        sir.stratify(age=['young', 'old']) # Labels, so not the age that people have
        sir.stratify(location=['urban', 'rural'], split=[0.7, 0.3])
        return sir
    mixing = ss.contacts(age=[[0.2, 0.3], [0.5, 0.7]], location=[[0.8, 0.2], [0.2, 0.8]])
    sims = sc.objdict({solver:run(make(), solver, networks=mixing, total_pop=10_000, dt=0.1, stop=20) for solver in solvers})
    res = sims.ode.results.sir
    assert sims.ode.strats[0].kind == 'plain'
    attack = {(a,l):res.n_R.select(age=a, location=l)[-1]/res.n_S.select(age=a, location=l)[0] for a in ['young', 'old'] for l in ['urban', 'rural']}
    assert attack['old', 'urban'] > attack['young', 'urban'] > attack['young', 'rural'] # Old people have more contacts, and most contacts are with one's own kind of place
    for solver in ['binomial', 'abm']:
        assert np.isclose(sims[solver].results.sir.n_R[-1], res.n_R[-1], rtol=0.1), f'Expected {solver} to match the ODE'
    assert sims.abm.results.sir.n_I[0] <= 10 # As agents, the number who start in each state is exact (less anyone who has already recovered)
    with pytest.raises(ss.CodedError, match='E603'):
        ss.Sim(make(), networks=ss.contacts(location=np.ones((3, 3))), **kw).init()

    # Where people start can be given for each level of a dimension
    def start(solver):
        sir = make_towns()
        sir.graph.states.I.init = ss.by(patch=[0, 500, 0])
        return run(sir, solver, total_pop=100_000, unit='day', stop=5)
    for solver in solvers:
        n_I = start(solver).results.sir.n_I.by('patch').values[0]
        assert n_I[0] == 0 and n_I[2] == 0 and 400 < n_I[1] <= 500

    # Age groups given as ages age by default, and do not if the sim says not to
    youngest = dict()
    for use_aging in [None, True, False]:
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), name='sir').stratify(age=[0, 5, 20]) # Nobody is infected, so everyone stays susceptible
        youngest[use_aging] = ss.Sim(sir, start=2000, stop=2010, use_aging=use_aging, **kw).run().results.sir.n_S.by('age').values[-1, 0]
    assert youngest[False] > 2*youngest[True] and youngest[None] == youngest[True]

    # A split over two dimensions at once, a value for every combination of levels, and a contact matrix within each place
    def make_places(mobility=None):
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.05)), recovery=ss.progression('I -> R', dur=ss.days(10)),
                         init=dict(I=ss.by(pd.DataFrame(dict(age=['0-4'], place=['a'], value=[10])))), name='sir')
        sir.stratify(age=[0, 5, 50])
        sir.stratify(place=['a', 'b'], kind='space', split=ss.by([[3000, 1000], [5000, 2000], [2000, 500]], dims=['age', 'place']), mobility=mobility)
        return ss.Sim(sir, networks=ss.contacts([[12, 4, 1], [4, 8, 2], [1, 2, 3]], age=[0, 5, 50]), total_pop=13_500, unit='day', stop=100, **kw)
    alone = make_places().run().results.sir
    n = [sum(alone[f'n_{k}'].select(age=age, place=place)[0] for k in 'SIR') for age in ['0-4', '50+'] for place in ['a', 'b']]
    assert np.allclose(n, [3000, 1000, 2000, 500])
    assert alone.n_I.select(age='0-4', place='a')[0] == 10 and alone.n_I[0] == 10
    assert alone.cum_infection.by('place').values[-1, 1] == 0 and alone.cum_infection[-1] > 5000 # Nobody moves, so the epidemic stays in a
    many = make_places().new(solver='binomial').run(n_runs=3).results.sir # Many runs keep their values by group too
    assert many.n_I.select(age='0-4', place='a').values.shape == (3, 101) and np.all(many.cum_infection.by('place')['b'].values == 0)
    moving = make_places(ss.Mobility(matrix=[[0, 0.01], [0.01, 0]], kind='migrate'))
    assert moving.run().results.sir.cum_infection.by('place').values[-1, 1] > 1000
    agents = make_places().new(n_agents=13_500).run()
    shares = [np.mean((agents.people.place[agents.people.age < 5] == 0)), np.mean((agents.people.place[agents.people.age >= 50] == 0))]
    assert np.allclose(shares, [0.75, 0.8], atol=0.05) # As agents, where people live depends on their age
    return sims


if __name__ == '__main__':
    mixing  = test_mixing()
    states  = test_states()
    strains = test_strains()
    space   = test_space()
    register = test_register()
