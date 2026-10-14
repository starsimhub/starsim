"""
Test dimensions that are more than a property of people: states with dimensions of their own, strains, and places
"""

# %% Imports and settings
import pytest
import numpy as np
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
    assert 'patch (space): north, south, east' in text and 'migrate' in text

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

    # Age groups given as ages do not age if the sim says not to
    youngest = dict()
    for use_aging in [True, False]:
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), name='sir').stratify(age=[0, 5, 20]) # Nobody is infected, so everyone stays susceptible
        youngest[use_aging] = ss.Sim(sir, start=2000, stop=2010, use_aging=use_aging, **kw).run().results.sir.n_S.by('age').values[-1, 0]
    assert youngest[False] > 2*youngest[True]
    return sims


if __name__ == '__main__':
    mixing  = test_mixing()
    states  = test_states()
    strains = test_strains()
    space   = test_space()
    register = test_register()
