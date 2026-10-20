"""
Test dimensions that are more than a property of people: states with dimensions of their own, strains, and places
"""

# %% Imports and settings
import math
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sps
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
        mortality  = Sv + Ev + Iv >> ss.null @ ss.days(10),
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
        kind       = 'strain',
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
        init_prev = 0, # Seeded by seed_north()
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
    assert np.allclose(res.n_Sv.by('species').values[0], [37_500, 12_500], rtol=0.1) # Split by species (after the first day, in which funestus dies twice as fast) ...
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

    with pytest.raises(ss.UnknownState):
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
    res = at_start(ss.Sim(seir, total_pop=1000, stop=2, **kw)).seir
    assert np.isclose(res.n_I.select(clinical='mild').values[0], 10) and np.isclose(res.n_I.values[0], 10)
    assert np.isclose(res.n_S.values[0], 990)

    with pytest.raises(ss.InvalidAdjustment): # People in E have no clinical level, so an adjustment of their onset cannot apply
        ss.Sim(clinical(adjust={'onset': [1, 2]}), **kw).init()

    # Infectiousness by clinical level: none at all stops transmission, and the solvers agree on the rest
    assert ss.Sim(clinical(adjust={'rel_trans': [0, 0]}), stop=100, **kw).run().results.seir.n_R.values[-1] < 101
    sizes = [ss.Sim(clinical(adjust={'rel_trans': [0.5, 2]}), solver=solver, stop=150, total_pop=20_000, **agents, **kw).run().results.seir.n_R.values[-1]
             for solver,agents in [('ode', {}), ('abm', dict(n_agents=20_000))]]
    assert np.isclose(*sizes, rtol=0.05)
    with pytest.raises(ss.InvalidAdjustment): # Susceptible people have no clinical level
        ss.Sim(clinical(adjust={'rel_sus': [1, 2]}), **kw).init()
    with pytest.raises(ss.MissingDimension): # Nor is there a level to take a value by clinical level from (before, it took the first)
        seir = ss.Disease(infection=ss.transmission('S -> E', beta=0.4, source='I'), onset=ss.progression('E -> I', rate=ss.by(clinical=[0.2, 0.4])), recovery=ss.progression('I -> R', dur=7), init=dict(I=0.01), name='seir')
        seir.stratify(clinical=['mild', 'severe'], states=['I'], split=[0.8, 0.2])
        ss.Sim(seir, **kw).init()
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
    by_strain = res.infection.cum.by('resistance').values[-1]
    assert res.n_active.by('resistance').values[0, 1] < 0.01 # Nobody starts with a resistant strain (so after the first step there is almost nobody)
    assert res.acquired.cum[-1] > 0 and by_strain[1] > 0 # It is acquired during treatment, and then transmitted
    assert by_strain[1] < 0.1*by_strain[0]
    for solver in ['binomial', 'abm']:
        other = sims[solver].results.tb
        assert np.isclose(other.infection.cum[-1], res.infection.cum[-1], rtol=0.05), f'Expected {solver} to match the ODE'
        assert np.isclose(other.n_active.by('resistance').values[-1, 0], res.n_active.by('resistance').values[-1, 0], rtol=0.1)
        assert 0 < other.infection.cum.by('resistance').values[-1, 1] < 0.1*by_strain[0]

    # Without acquired resistance, the resistant strain never appears; and a fitter strain takes over
    def make(adjust):
        sis = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> S', dur=10), init_prev=0.01, name='sis')
        return sis.stratify(strain=['wild', 'variant'], kind='strain', split=[0.9, 0.1], adjust={'infection': adjust})
    for solver in solvers:
        res = run(make([1.0, 1.5]), solver, stop=300, total_pop=50_000).results.sis
        n_I = res.n_I.by('strain').values
        assert n_I[0, 0] > 5*n_I[0, 1] and n_I[-1, 1] > 5*n_I[-1, 0], f'Expected the more transmissible strain to take over with {solver}'
    # Cross-immunity between strains: complete protection is the base model; none lets each person have each strain (as agents, who keep every strain they have had)
    def flu(immunity):
        dis = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=7), init_prev=0.01, name='flu')
        return dis.stratify(strain=['A', 'B'], kind='strain', immunity=immunity)
    total = lambda immunity, **kwargs: ss.Sim(flu(immunity), unit='day', stop=200, **kwargs, **kw).run().results.flu.infection.sum()
    once = total(None)
    assert np.isclose(total(np.ones((2, 2))), once)
    twice = total(np.eye(2), n_agents=10_000)
    assert 1.5*once < twice < 2.2*once
    with pytest.raises(ss.CapabilityError): # A compartmental run would keep only the last strain, so people could go from A to B and back (KI15) ...
        total(np.eye(2), solver='ode')
    assert np.isclose(total(np.eye(2)), twice, rtol=0.1) # ... so partial immunity runs as agents
    assert total(ss.immunity(rel_sus=np.eye(2)), n_agents=10_000) == twice # A bare matrix is ss.immunity with the default rule, the strongest
    for combine in ['sum', 'multiply', lambda p: p.max(0)]: # With protection from one strain only, the rules agree
        assert total(ss.immunity(rel_sus=np.eye(2), combine=combine), n_agents=10_000) == twice
    with pytest.raises(ss.CapabilityError): # A function has no compartmental form (nor a gradient)
        total(ss.immunity(rel_sus=np.ones((2, 2)), combine=lambda p: p.max(0)), solver='ode')
    with pytest.raises(ValueError): # In a dimension, a person is protected by the strains they have had, against infection
        flu(ss.immunity(protected='R', rel_sus=np.eye(2)))
    with pytest.raises(ss.ShapeError):
        flu(np.eye(3))

    return sims


@sc.timer()
def test_carriage():
    sc.heading('Testing strains carried together: superinfection, and a transition that changes the strains carried')
    def make():
        S, L, I = ss.states(3)
        dis = ss.Disease(
            infection      = S >> L @ ss.peryear(6)*I,
            superinfection = L >> L @ ss.peryear(3)*I, # Someone with A who is infected with B carries both
            activation     = L >> I @ ss.peryear(0.5),
            to_A           = ss.progression('L -> I', rate=ss.peryear(0.3), move=dict(strain='A+B -> A')), # Only one of the two progresses
            to_B           = ss.progression('L -> I', rate=ss.peryear(0.2), move=dict(strain='A+B -> B')),
            recovery       = I >> S @ ss.peryear(1),
            init           = dict(I=0.02),
            name           = 'dis',
        )
        return dis.stratify(strain=['A', 'B', 'A+B'], kind='strain', split=[0.5, 0.5, 0], adjust={'activation': [1, 1, 0.4]})

    # The reference: someone with A and B passes on each with half their infectiousness
    N = 100_000
    def deriv(t, y):
        S, LA, LB, LAB, IA, IB, IAB = y
        la, lb = 6*(IA + IAB/2)/N, 6*(IB + IAB/2)/N
        return [-(la + lb)*S + IA + IB + IAB, la*S - lb/2*LA - 0.5*LA, lb*S - la/2*LB - 0.5*LB, lb/2*LA + la/2*LB - 0.7*LAB,
                0.5*LA + 0.3*LAB - IA, 0.5*LB + 0.2*LAB - IB, 0.2*LAB - IAB]
    from scipy.integrate import solve_ivp
    ref = solve_ivp(deriv, [0, 20], [0.98*N, 0, 0, 0, 0.01*N, 0.01*N, 0], rtol=1e-10, atol=1e-6).y[:, -1]
    for solver in solvers:
        res = run(make(), solver, start=2000, stop=2020, dt=ss.years(0.02), total_pop=N).results.dis
        n_L, n_I = res.n_L.by('strain').values[-1], res.n_I.by('strain').values[-1]
        rtol = 1e-3 if solver == 'ode' else 0.15
        assert np.allclose(n_L, ref[1:4], rtol=rtol) and np.allclose(n_I, ref[4:], rtol=rtol), f'Expected {solver} to match the reference: {n_L}, {n_I} vs {ref[1:]}'
    with pytest.raises(ss.ShapeError):
        ss.transitions.Strat('strain', ['A', 'A+C'], kind='strain') # C has no level of its own
    return res


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
            attack = res.infection.cum.by('patch').values[-1]/[50_000, 30_000, 20_000]
            if key == 'none': # People mix within their own town, so the epidemic stays where it started
                assert attack[0] > 0.9 and np.allclose(attack[1:], 0)
            else:
                assert np.all(attack > 0.85), f'Expected the epidemic to reach every town with {key} and {solver}, not {attack}'
            if solver == 'abm': # Each agent's place is its group too: one 16-bit array, not three
                assert sim.people.patch.raw.dtype == np.int16 and sim.groups.group_raw() is sim.people.patch.raw
            peaks[f'{key}_{solver}'] = res.n_I.by('patch').values.argmax(axis=0)
    assert peaks.gravity_ode[0] < peaks.gravity_ode[1] < peaks.gravity_ode[2] # The nearer town is reached first
    assert np.ptp(peaks.away_ode) < 3 < np.ptp(peaks.gravity_ode) # The more time people spend away, the closer together the towns peak
    text = sim.describe(output=True)
    assert 'patch (space): north, south, east' in text and "kind='force'" in text

    # The kind of a dimension comes from its value, not its name: places are an ss.geo (or kind='space'), and a name such as region is noted
    def regions(levels):
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init=dict(I=ss.by(region=[10, 0])), name='sir')
        return ss.Sim(sir.stratify(region=levels), total_pop=20_000, unit='day', stop=200, solver='ode', **kw).run()
    plain, places = regions(['a', 'b']), regions(ss.geo(names=['a', 'b']))
    assert np.all(plain.results.sir.infection.cum.by('region').values[-1] > 4000) and any('"region" is plain' in note for note in plain.notes) # Its levels mix
    assert places.results.sir.infection.cum.by('region').values[-1, 1] == 0 # Places are apart

    # A moving force of infection goes only where the matrix sends it: from a to b, not from b to a
    for matrix,reached in [([[0, 0.1], [0, 0]], True), ([[0, 0], [0.1, 0]], False)]:
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init=dict(I=ss.by(place=[10, 0])), name='sir')
        sir.stratify(place=['a', 'b'], kind='space', mobility=ss.Mobility(matrix=matrix, kind='force'))
        for solver in ['ode', 'abm']:
            attack = run(sir, solver, total_pop=20_000, unit='day', stop=200).results.sir.infection.cum.by('place').values[-1]
            assert attack[0] > 5000 and (attack[1] > 5000) == reached, f'Expected b to be reached: {reached}, with {solver}, not {attack}'

    # A movement matrix may be sparse, for many places; the time share is kept sparse either way, with the same run
    finals = []
    for matrix in [[[0, 0.1], [0.05, 0]], sps.csr_array([[0, 0.1], [0.05, 0]])]:
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init=dict(I=ss.by(place=[10, 0])), name='sir')
        sim = run(sir.stratify(place=['a', 'b'], kind='space', mobility=ss.Mobility(matrix=matrix, kind='commute')), 'ode', total_pop=20_000, unit='day', stop=100)
        assert sps.issparse(sim.mixing.time_share)
        finals.append(sim.results.sir.infection.cum.by('place').values[-1])
    assert np.array_equal(*finals) and finals[0][1] > 1000 # The same with either

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
    cholera.stratify(patch=['upstream', 'downstream'], kind='space', adjust={'shedding': [1.0, 0.0]})
    for solver in ['ode', 'abm']:
        res = run(cholera, solver, total_pop=100_000, unit='day', stop=200).results.cholera
        assert res.n_W.by('patch').values[:, 1].max() == 0 # Nobody sheds downstream ...
        assert res.infection.cum.by('patch').values[-1, 1] == 0 # ... so nobody is infected there
        assert res.infection.cum.by('patch').values[-1, 0] > 40_000

    # The passes over the agents are split between threads for a large population (here, for a small one): the results are those of one thread
    defaults = ss.kernels.min_threaded, ss.kernels.chunk_size
    ss.kernels.min_threaded, ss.kernels.chunk_size = 0, 1000
    try:
        sims = []
        for threads in [1, 0]: # One thread, then as many as Numba has
            with ss.options.context(threads=threads):
                sim = ss.Sim(make_towns(mobility=ss.gravity(k=2e-4)), stop=60, unit='day', n_agents=20_000, total_pop=100_000, **kw).init()
                sim.loop.insert(seed_north, label='sim.start_step')
                sims.append(sim.run())
    finally:
        ss.kernels.min_threaded, ss.kernels.chunk_size = defaults
    infections = [sim.results.sir.infection.values for sim in sims]
    assert infections[0].sum() > 0 and np.array_equal(*infections) # The same people infected, with threads or without

    # The share of time away can be fitted: the mixing is worked out again for each value tried
    def commuters(away):
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init=dict(I=ss.by(patch=[10, 0, 0])), name='sir')
        return sir.stratify(patch=ss.geo(names=['a', 'b', 'c'], pop=[1e5, 1e5, 1e5], x=[0, 100, 200], y=[0, 0, 0]), mobility=ss.Mobility(ss.gravity(), away=away))
    pars = dict(total_pop=3e5, unit='day', stop=150, **kw)
    for solver in ['ode', 'binomial']:
        truth = ss.Sim(commuters(0.02), solver=solver, **pars).run().results.sir.infection
        data = sc.dataframe(date=truth.timevec, **{f'infection[patch={p}]': truth.select(patch=p) for p in 'abc'})
        away = ss.par(0.1, bounds=[0.002, 0.3])
        fit = ss.Sim(commuters(away), data=data, solver=solver, **pars).calibrate(verbose=0)
        rtol = 0.05 if solver == 'ode' else 0.2 # A stochastic fit takes fresh seeds at each point, so it does not reproduce the data's own draws
        assert np.isclose(fit.pars.away, 0.02, rtol=rtol), f'Expected to recover away=0.02 with {solver}, not {fit.pars.away}' # Before, a TypeError at init
    assert 'away=away' in ss.Sim(commuters(away), **pars).to_code() # The saved form keeps the fitted parameter, with its bounds, not its value
    sim = ss.Sim(commuters(away), n_agents=3000, solver='abm', **pars).init()
    shares = [sim.new(pars=dict(away=val)).init().mixing.time_share.diagonal() for val in [0.01, 0.2]] # As agents too: the share of time at home
    assert np.allclose(shares[0], 0.99) and np.allclose(shares[1], 0.8)
    with pytest.raises(ValueError, match='sim.new'): # Rather than restarting with the old mixing
        ss.Sim(commuters(away), **pars).restart(pars=dict(away=0.05))

    # A gravity model needs a geography
    with pytest.raises(ss.NoGeography):
        sir = ss.Disease(recovery=ss.progression('I -> R', dur=3)).stratify(patch=['a', 'b'], mobility=ss.gravity())
        ss.Sim(sir, **kw).init()

    # Newborns from ss.Births, written as code, are placed by the split, as declared births are (before, all in the first place)
    sis = ss.Disease(infection=ss.transmission('S -> I', beta=ss.peryear(0.5)), recovery=ss.progression('I -> S', dur=ss.years(1)), init_prev=0.1, name='sis')
    sis.stratify(patch=['a', 'b', 'c'], kind='space', split=[0.4, 0.3, 0.3])
    sim = ss.Sim(sis, n_agents=5_000, demographics=[ss.Births(birth_rate=ss.peryear(40)), ss.Deaths(death_rate=ss.peryear(40))], start=2000, stop=2020, verbose=0).run()
    n = sim.results.sis.n_S.by('patch').values[-1] + sim.results.sis.n_I.by('patch').values[-1]
    assert np.allclose(n/n.sum(), [0.4, 0.3, 0.3], atol=0.03) # Was 0.73, 0.13, 0.13
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
    assert sims.ode.results.sir.infection.cum.by('place').values[-1, 1] > 1000 # People moved from a took the epidemic to b

    # Stepping gives what the whole run in one call does; a date is placed on the sim's calendar; a whole number is a count of people
    with pytest.warns(RuntimeWarning):
        stepped = ss.Sim(make(), solver='ode', total_pop=10_000, unit='day', stop=60, **kw).run(profile=True)
    assert np.allclose(stepped.results.sir.n_I.values, sims.ode.results.sir.n_I.values)
    dated = register.copy()
    dated['date'] = [ss.date('2020-01-01') + ss.days(int(d)) for d in register['date']]
    with pytest.warns(RuntimeWarning):
        sim = ss.Sim(make(dated), solver='ode', total_pop=10_000, unit='day', start='2020-01-01', stop='2020-03-01', **kw).run()
    assert np.allclose(sim.results.sir.n_S.values, sims.ode.results.sir.n_S.values)
    res = at_start(ss.Sim(make(register.iloc[:1], init=ss.by(place=[1, 0, 0])), solver='ode', total_pop=10_000, unit='day', stop=5, **kw)).sir
    assert np.allclose(res.n_I.by('place').values[0], [0.8, 0.2, 0]) # One person in a (not all of them), of whom 1000 in 5000 moved to b

    # With another disease tracked jointly, in either order, everyone is moved once, by the states of the disease that has them (before, the events at the start were lost, and the first disease's states were used: 6% and 44% off)
    def crossed(other_first):
        sir = make()
        other = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.2)), recovery=ss.progression('I -> S', dur=ss.days(7)), init_prev=0.02, name='other')
        same = ss.modify(sir.rel_sus, multiply=1.0, where=other.I, name='same') # Changes nothing, but the diseases are tracked jointly
        return ss.Sim(diseases=[other, sir] if other_first else [sir, other], connectors=same, solver='ode', total_pop=10_000, unit='day', stop=60, **kw)
    for other_first in [False, True]:
        with pytest.warns(RuntimeWarning):
            res = crossed(other_first).run().results.sir
        assert np.allclose(res.n_I.values, sims.ode.results.sir.n_I.values, rtol=1e-4) # To the ODE's accuracy: the other disease's growth can give the joint run more substeps

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
    assert attack['old', 'urban'] > attack['young', 'urban'] # Old people have more contacts
    assert np.isclose(attack['young', 'urban'], attack['young', 'rural']) # The two kinds of place mix alike and are seeded as their people are split, so they end alike (before, a strict > held by a rounding error)
    for solver in ['binomial', 'abm']:
        assert np.isclose(sims[solver].results.sir.n_R[-1], res.n_R[-1], rtol=0.1), f'Expected {solver} to match the ODE'
    assert sims.abm.results.sir.n_I[0] <= 10 # As agents, the number who start in each state is exact (less anyone who has already recovered)
    with pytest.raises(ss.ShapeError):
        ss.Sim(make(), networks=ss.contacts(location=np.ones((3, 3))), **kw).init()
    assert mixing.pars.n_contacts[1, 2] == 0.5*0.2 # From (old, urban) to (young, rural)
    T = sc.timer()
    ss.Sim(make(), networks=ss.contacts(place=np.eye(500) + 0.01), **kw) # A matrix over many places is copied into the sim as numbers, not as 250,000 objects
    assert T.toc(output=True) < 5 # About 0.1 s; 774 places took 170 s

    # Where people start can be given for each level of a dimension
    def start(solver):
        sir = make_towns()
        sir.graph.states.I.init = ss.by(patch=[0, 500, 0])
        return run(sir, solver, total_pop=100_000, unit='day', stop=5)
    for solver in solvers:
        n_I = start(solver).results.sir.n_I.by('patch').values[0]
        assert n_I[0] == 0 and n_I[2] == 0 and 400 < n_I[1] < 700 # After the first day

    # People age between age groups only if they are born or die, unless the sim says otherwise
    youngest = dict()
    for use_aging in [None, True, False]:
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), name='sir').stratify(age=[0, 5, 20]) # Nobody is infected, so everyone stays susceptible
        youngest[use_aging] = ss.Sim(sir, start=2000, stop=2010, use_aging=use_aging, **kw).run().results.sir.n_S.by('age').values[-1, 0]
    assert youngest[False] > 2*youngest[True] and youngest[None] == youngest[False] # A closed population does not age by default
    born = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), births=ss.birth('-> S', rate=ss.peryear(0.02)), deaths=ss.death(rate=ss.peryear(0.02)), name='sir').stratify(age=[0, 5, 20])
    assert ss.Sim(born, start=2000, stop=2001, **kw).init().pars.use_aging # With births and deaths, people age

    # A split over two dimensions at once, a value for every combination of levels, and a contact matrix within each place
    def make_places(mobility=None):
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.05)), recovery=ss.progression('I -> R', dur=ss.days(10)),
                         init=dict(I=ss.by(pd.DataFrame(dict(age=['0-4'], place=['a'], value=[10])))), name='sir')
        sir.stratify(age=[0, 5, 50])
        sir.stratify(place=['a', 'b'], kind='space', split=ss.by([[3000, 1000], [5000, 2000], [2000, 500]], dims=['age', 'place']), mobility=mobility)
        return ss.Sim(sir, networks=ss.contacts([[12, 4, 1], [4, 8, 2], [1, 2, 3]], bands=[0, 5, 50]), total_pop=13_500, unit='day', stop=100, **kw)
    start = at_start(make_places()).sir
    n = [sum(start[f'n_{k}'].select(age=age, place=place)[0] for k in 'SIR') for age in ['0-4', '50+'] for place in ['a', 'b']]
    assert np.allclose(n, [3000, 1000, 2000, 500])
    assert start.n_I.select(age='0-4', place='a')[0] == 10 and start.n_I[0] == 10
    alone = make_places().run().results.sir
    assert alone.infection.cum.by('place').values[-1, 1] == 0 and alone.infection.cum[-1] > 5000 # Nobody moves, so the epidemic stays in a
    many = make_places().new(solver='binomial').run(n_runs=3).results.sir # Many runs keep their values by group too
    assert many.n_I.select(age='0-4', place='a').values.shape == (3, 101) and np.all(many.infection.cum.by('place')['b'].values == 0)
    moving = make_places(ss.Mobility(matrix=[[0, 0.01], [0.01, 0]], kind='migrate'))
    assert moving.run().results.sir.infection.cum.by('place').values[-1, 1] > 1000
    agents = make_places().new(n_agents=13_500).run()
    shares = [np.mean((agents.people.place[agents.people.age < 5] == 0)), np.mean((agents.people.place[agents.people.age >= 50] == 0))]
    assert np.allclose(shares, [0.75, 0.8], atol=0.05) # As agents, where people live depends on their age

    # As agents, the contact matrix acts within each place too, with commuting between them
    assert agents.results.sir.infection.cum.by('place').values[-1, 1] == 0
    commute = ss.Mobility(matrix=[[0.9, 0.1], [0.1, 0.9]], kind='commute')
    ode, abm = [make_places(commute).new(**pars).run().results.sir.infection.cum.by('place').values[-1] for pars in [{}, dict(n_agents=13_500)]]
    assert np.allclose(abm, ode, rtol=0.05) and ode[1] > 1000

    # So do a network and a mixing pool of everyone: as agents, contacts are within each place, or where people commute to
    def make_net(net, mobility=None, **pars):
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.03)), recovery=ss.progression('I -> R', dur=ss.days(10)), init=dict(I=ss.by(patch=[20, 0, 0])), name='sir')
        sir.stratify(patch=['a', 'b', 'c'], kind='space', split=[0.5, 0.3, 0.2], mobility=mobility)
        return ss.Sim(sir, networks=net, total_pop=30_000, unit='day', stop=120, **kw, **pars)
    commute = ss.Mobility(matrix=[[0, 0.05, 0], [0.05, 0, 0.05], [0, 0.05, 0]], kind='commute')
    for net in [ss.RandomNet(n_contacts=10, dur=0, name='net'), ss.MixingPool(n_contacts=10, name='net')]:
        alone = make_net(net, n_agents=30_000).run().results.sir.infection.cum.by('patch').values[-1]
        assert alone[0] > 10_000 and alone[1] == alone[2] == 0
        ode, abm = [make_net(sc.dcp(net), commute, **pars).run().results.sir.infection.cum.by('patch').values[-1] for pars in [dict(solver='ode', approx=True), dict(n_agents=30_000)]]
        assert np.allclose(abm, ode, rtol=0.05) and ode[2] > 1000

    # An explicit contact matrix over the places is used whatever the dimension is called
    for solver in ['ode', 'abm']:
        final = {dim:corpus.patch_matrix(dim=dim, solver=solver, verbose=0).run().results.sir.n_R.by(dim).values[-1] for dim in ['patch', 'zone']}
        assert np.allclose(final['patch'], final['zone'], rtol=0.01) and final['patch'][1] > 9000 # Not [9,802, 0] with patch

    # A contact matrix is aggregated to the model's age groups by population, so the ODE matches agents, who have ages
    def attack(solver):
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.03), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir').stratify(age=[0, 50])
        sim = ss.Sim(sir, networks=ss.contacts([[12, 1, 1], [1, 2, 1], [1, 1, 1]], bands=[0, 5, 50]), people=ss.people('Kenya'), use_aging=False, unit='day', stop=300, solver=solver, **kw)
        res = sim.run().results.sir
        return res.infection.cum[-1]/res.n_S[0]
    assert np.isclose(attack('ode'), attack('abm'), rtol=0.05) # 0.58 and 0.57; 0.50 by band width
    S, I, R = ss.states('S I R') # A count in a rate is used literally, so the contacts do not apply to it, which is noted
    literal = ss.Disease(infection=S >> I @ 0.05*I.n/(S + I + R).n, recovery=I >> R @ ss.days(10), init=dict(I=0.01), name='sir').stratify(age=[0, 15])
    assert any('literally' in note for note in ss.Sim(literal, networks=ss.contacts([[3, 1], [1, 2]], bands=[0, 15]), **kw).init().notes)
    for bad in [dict(matrix=[[-5, 0], [0, 1]], bands=[0, 20]), dict(matrix=[[1, 0], [0, 1]], bands=[20, 0]), dict(age=[[1, np.nan], [1, 1]])]:
        with pytest.raises(ValueError):
            ss.contacts(**bad)

    # A split over two dimensions is the share in each level given the other, whose own split, if given, is kept (FPsim review: it was replaced by the table's totals)
    for solver in ['ode', 'abm']:
        d = ss.Disease(ss.State('A'), stay=ss.progression('A -> A', rate=0), name='d')
        d.stratify(k=['x', 'y'], split=ss.by([[0.8, 0.2], [0.4, 0.6]], dims=['age', 'k']))
        size = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
        n_A = at_start(ss.Sim(d, solver=solver, stratify=dict(age=[0, 30], split=[0.9, 0.1]), stop=1, **size, **kw)).d.n_A
        assert np.allclose(n_A.by('age').values[0], [9000, 1000], rtol=0.03)
        assert np.allclose([n_A.select(age=a, k='y')[0] for a in ['0-29', '30+']], [1800, 600], rtol=0.1)

    # Where people start, given at levels where the split puts nobody, is an error, not zero (FPsim review: everyone starts at ever=False)
    def make_init(table):
        A, B = ss.states('A B')
        d = ss.Disease(A, B, ab=A >> B @ ss.peryear(0), init=dict(B=ss.by(np.array(table), dims=['age', 'ever'])), name='d')
        d.stratify(ever=[False, True], split=[1, 0])
        return ss.Sim(d, total_pop=1000, stratify=dict(age=[0, 30]), stop=1, **kw)
    with pytest.raises(ss.InitWhereNobodyIs, match='ever=True'):
        make_init([[0, 0.5], [0, 0.5]]).init()
    assert any('ever=True' in note for note in make_init([[0, 0.5], [0.2, 0.5]]).init().notes) # Some of it: a note
    return sims


@sc.timer()
def test_ragged():
    sc.heading('Testing a dimension for some groups only: age groups for women')
    def make():
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.2), recovery=ss.progression('I -> R', dur=10), init=dict(I=ss.by(age=[0.004, 0.002, 0.001, 0.0])), name='sir')
        sir.stratify(sex=['female', 'male'])
        sir.stratify(age=[15, 25, 35, 50], where=dict(sex='female'), adjust=dict(recovery=[2, 1, 1, 0.5]))
        return sir
    sims = sc.objdict({solver:run(make(), solver, unit='day', stop=150, total_pop=100_000) for solver in solvers})
    ode = sims.ode
    assert np.all(ode.group_levels[ode.group_coords[:, 0] == 1, 1] == -1) # Men have no age group
    res = ode.results.sir
    start = at_start(ss.Sim(make(), unit='day', stop=150, total_pop=100_000, **kw)).sir
    split = next(strat for strat in ode.strats if strat.name == 'age').split
    assert np.isclose(start.n_I.select(sex='male')[0], 50_000*(split @ [0.004, 0.002, 0.001, 0.0]), rtol=0.01) # A value for each age group is, for men, the mean over them, weighted by the split
    assert np.allclose(res.n_R.by('age').values.sum(axis=1), res.n_R.select(sex='female').values) # Selecting an age group selects women only
    for solver in ['binomial', 'abm']:
        other = sims[solver].results.sir
        assert np.allclose(other.n_R.by('sex')[-1], res.n_R.by('sex')[-1], rtol=0.05), f'Expected {solver} to match the ODE'
        assert np.allclose(other.n_R.by('age')[-1], res.n_R.by('age')[-1], rtol=0.1), f'Expected {solver} to match the ODE'

    # A condition on the dimension leaves out the people who have no level of it
    for solver in ['ode', 'abm']:
        sir = make()
        sim = run(sir, solver, unit='day', stop=150, total_pop=100_000, interventions=ss.modify(sir.rel_sus, multiply=0, where=sir.age == '15-24'))
        r = sim.results.sir
        assert r.infection.cum.select(age='15-24')[-1] == 0 and r.infection.cum.select(sex='male')[-1] > 10_000

    # A dimension within a ragged one, set on the sim; contacts by a ragged dimension have no level for the others to mix as
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.2), recovery=ss.progression('I -> R', dur=10), init=dict(I=0.001), name='sir')
    sim = ss.Sim(sir, stratify=[dict(sex=['female', 'male']), dict(age=[15, 25, 35, 50], where=dict(sex='female')), dict(parity=['0', '1+'], split=[0.4, 0.6], where=dict(age=['25-34', '35-49']))], total_pop=10_000, unit='day', stop=50, **kw).run()
    split = next(strat for strat in sim.strats if strat.name == 'age').split
    assert (sim.group_levels[:, 2] >= 0).sum() == 4 and np.isclose(sim.results.sir.n_S.by('parity').values[0].sum(), 5_000*split[1:3].sum(), rtol=0.01) # Women aged 25 to 49
    with pytest.raises(ss.PartialStratificationMixing):
        ss.Sim(make(), networks=ss.contacts([[3, 1, 1, 1], [1, 3, 1, 1], [1, 1, 3, 1], [1, 1, 1, 3]], bands=[15, 25, 35, 50]), **kw).init()
    with pytest.raises(ss.PartialStratificationMixing): # So do contacts by a dimension that the infectious state does not have
        ss.Sim(sir.copy().stratify(age=[0, 15], states=['S', 'R']), networks=ss.contacts([[3, 1], [1, 3]], bands=[0, 15]), **kw).init()
    with pytest.raises(ss.MissingDimension):
        ss.Sim(sir.copy().stratify(age=[0, 15], where=dict(sex='female')), **kw).init()

    # As agents, a sex dimension is people.female, which a sexual network pairs on
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.5), recovery=ss.progression('I -> R', dur=30), init_prev=0.05, name='sir')
    sir.stratify(sex=['female', 'male'], split=[0.6, 0.4], adjust={'infection': [0, 1]})
    sim = ss.Sim(sir, networks=ss.MFNet(), n_agents=5_000, unit='day', stop=50, **kw).run()
    res, women = sim.results.sir, sim.people.female.uids
    assert np.isclose(len(women), 3_000, rtol=0.05) and np.all(sim.people.sex[women] == 0) # The split is the share of women
    assert sim.diseases.sir.I[women].sum() + sim.diseases.sir.R[women].sum() == res.n_I.select(sex='female')[-1] + res.n_R.select(sex='female')[-1] # Not 51 women infected through the network and 0 in the results
    return sims


def test_sex():
    """ As agents, a dimension of sex is people.female, for everyone including newborns """
    sc.heading('Testing sex as a dimension of agents')
    for kw in [dict(sex=['female', 'male']), dict(sex=['m', 'f']), dict(sex=['female', 'male'], split=[0.7, 0.3])]:
        si = ss.Disease(infection=ss.transmission('S -> I', beta=0.01), init_prev=0.05, name='si')
        sim = ss.Sim(si, networks=ss.MFNet(), demographics=ss.Births(birth_rate=30), n_agents=2000, stratify=kw, stop=2010, verbose=0, rand_seed=1).init()
        if 'split' in kw:
            assert np.isclose(sim.people.female.mean(), 0.7, atol=0.03) # people.female follows a split that is given (drawn, for 2000 agents)
        sim.run()
        ppl = sim.people
        strat = sim.strats[0]
        women = ppl[strat.attr].raw[ppl.auids] == strat.female
        assert np.array_equal(women, ppl.female.raw[ppl.auids]), f'{kw}: the levels of sex disagree with people.female'
        assert sim.results.si.n_S.select(**{strat.name: strat.labels[strat.female]})[-1] == (ppl.female & sim.diseases.si.S).sum()
    with pytest.raises(ValueError, match='people.female'):
        ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.01), name='si'), n_agents=100, stratify=dict(sex=['a', 'b']), verbose=0).init()
    return


@sc.timer()
def test_property_values():
    sc.heading('Testing that people hold the values of a numeric dimension, and keep a property they already have')
    # Numeric levels are lower bin edges, and people hold the value (FP19: they held 0 to 4, binned by the edges 1 to 5, so the top level was empty)
    S, I = ss.states('S I')
    sim = ss.Sim(ss.Disease(inf=S >> I @ ss.peryear(0.0)*I, name='d'), n_agents=5000, start=2000, stop=2001, solver='abm', verbose=0)
    sim.stratify(wealthquintile=[1, 2, 3, 4, 5])
    sim.run()
    assert set(np.unique(sim.people.wealthquintile[sim.people.auids])) == {1, 2, 3, 4, 5}
    res = sim.results.d
    assert np.allclose(res.n_S.by('wealthquintile').values[0] + res.n_I.by('wealthquintile').values[0], 1000)

    # A property people already have is kept and binned (FP20: urban at 0.9 became 0.000), and split= for it raises
    def make(**kw):
        ppl = ss.People(5000, extra_states=[ss.BoolState('urban', default=ss.bernoulli(0.9)), ss.FloatArr('wealth', default=ss.uniform(0, 100))])
        S, I = ss.states('S I')
        sim = ss.Sim(ss.Disease(inf=S >> I @ ss.peryear(0.0)*I, name='d'), people=ppl, start=2000, stop=2001, solver='abm', verbose=0)
        sim.stratify(urban=[False, True], **kw)
        sim.stratify(wealth=[0, 50])
        return sim
    sim = make().run()
    assert np.isclose(sim.people.urban.mean(), 0.9, atol=0.02) and len(np.unique(sim.people.wealth[sim.people.auids])) > 100
    with pytest.raises(ss.BadSplit):
        make(split=[0.5, 0.5]).init()
    return sim


@sc.timer()
def test_properties_in_rates():
    sc.heading('Testing people\'s properties and conditions in rates, as a regression')
    from scipy.special import expit

    # A probability of use as a regression on a property (FP26), the same as agents and as compartments, and stopped by ss.change on its intercept (FP14: 3,531 starts with and without it)
    def make(solver, change=False):
        b0, b_urban = ss.par(-1.0, bounds=[-3, 1]), ss.par(1.0)
        start = ss.Transition('none', 'on', rate=ss.peryear(1)*expit(b0 + b_urban*ss.ppl.urban))
        meth = ss.Disease(start=start, stop=ss.Transition('on', 'none', rate=ss.peryear(0.5)), b0=b0, b_urban=b_urban, name='meth')
        intv = ss.change('meth.pars.b0', at=2005, to=-np.inf) if change else None
        sim = ss.Sim(meth, interventions=intv, n_agents=20_000, start=2000, stop=2010, dt=1/12, solver=solver, verbose=0)
        sim.stratify(urban=[False, True], split=[0.4, 0.6])
        return sim.run()
    on = [make(solver).results.meth.n_on[-1] for solver in ['abm', 'ode']]
    expected = 20_000*(0.4*expit(-1)/(expit(-1) + 0.5) + 0.6*0.5/(0.5 + 0.5)) # At equilibrium
    assert np.isclose(on[1], expected, rtol=0.01) and np.isclose(on[0], expected, rtol=0.03)
    for solver in ['abm', 'ode']:
        assert make(solver, change=True).results.meth.start[61:].sum() == 0

    # A condition in a rate is 1 or 0 for each person; on states alone joined by |, it is still an error
    S, I, R = ss.states('S I R')
    sir = ss.Disease(S, I, R, infection=S >> I @ 0.3*I, recovery=I >> R @ ss.peryear(1)*(1 + 2*(ss.ppl.age > 50)), init=dict(I=0.01), name='sir')
    sim = ss.Sim(sir, n_agents=10_000, stop=5, dt=0.1, use_aging=False, verbose=0).run()
    ode = ss.Sim(sir, total_pop=10_000, stop=5, dt=0.1, use_aging=False, solver='ode', verbose=0)
    ode.stratify(age=[0, 50])
    ode.run()
    assert np.isclose(sim.results.sir.n_R[-1], ode.results.sir.n_R[-1], rtol=0.15)
    with pytest.raises(TypeError, match=r'0.3\*\(Ia \+ Is\)'):
        ss.Disease(S, I, R, infection=S >> I @ 0.3*(S | I))

    # Written to the model spec as people.urban and expit(), and back as code with the same hash
    sim = make('abm')
    spec = sim.to_spec()
    assert 'expit(b0 + b_urban * people.urban)' in spec['modules'][0]['transitions'][0]['rate']
    code = ss.to_code(spec)
    assert 'from scipy.special import expit' in code and 'ss.ppl.urban' in code
    ns = {}
    exec(code.replace("if __name__ == '__main__':", 'if False:'), ns)
    assert ns['sim'].hash() == sim.hash()

    # A property that is not one of the people's says what is
    bad = ss.Disease(S, I, R, infection=S >> I @ 0.3*I, recovery=I >> R @ ss.peryear(1)*ss.ppl.agee, name='bad')
    with pytest.raises(ss.UnknownState, match='age'):
        ss.Sim(bad, n_agents=100, verbose=0).init()
    return sim


@sc.timer()
def test_by_properties():
    sc.heading('Testing values by a property of people that is not a dimension (D15), and tables with bin edges')
    def make(mu, solver='auto', dims=None, **kw):
        S, D = ss.states('S D')
        sim = ss.Sim(ss.Disease(S, D, death=S >> D @ mu, name='d'), solver=solver, use_aging=False, verbose=0, **sc.mergedicts(dict(stop=10), kw))
        for dim,levels in (dims or {}).items():
            sim.stratify(**{dim:levels})
        return sim

    # Death rates by sex with no sex dimension: read from people.female as agents; an explicit ODE says to stratify
    mu = ss.by(sex=dict(female=ss.peryear(0.02), male=ss.peryear(0.05)))
    sim = make(mu, n_agents=20_000).run()
    assert sim.solver == 'abm'
    female, dead = sim.people.female.raw[sim.people.auids], sim.diseases.d.D.raw[sim.people.auids]
    for sel,rate in [(female, 0.02), (~female, 0.05)]:
        p = 1 - np.exp(-10*rate)
        assert abs(dead[sel].mean() - p) < 3*np.sqrt(p*(1 - p)/sel.sum())
    with pytest.raises(ss.CapabilityError, match='stratify'):
        make(mu, solver='ode', total_pop=20_000).init()
    ode = make(mu, solver='ode', total_pop=20_000, dims=dict(sex=['female', 'male'])).run() # The same keys, by a dimension
    assert np.allclose(ode.results.d.n_D.by('sex').values[-1], 10_000*(1 - np.exp(-10*np.array([0.02, 0.05]))), rtol=1e-3)
    with pytest.raises(ss.MissingDimension, match=r'dict\(female='):
        make(ss.by(sex=[ss.peryear(0.02), ss.peryear(0.05)]), n_agents=100).init()
    with pytest.raises(ss.MissingDimension, match='lower edge'):
        make(ss.by(age=[ss.peryear(0.02), ss.peryear(0.05)]), n_agents=100).init()

    # A table with bin edges, by single years of age or coarser: each person's own age as agents; averaged over each age group as compartments, with a note
    table = ss.by(dict(age=[0, 20, 40, 60], value=[ss.peryear(v) for v in [0.01, 0.02, 0.03, 0.04]]))
    abm = make(table, n_agents=50_000, solver='abm').run()
    ode = make(table, total_pop=50_000, solver='ode', dims=dict(age=[0, 30, 60])).run()
    assert np.isclose(abm.results.d.n_D[-1], ode.results.d.n_D[-1], rtol=0.03)
    assert any('averages it over each group' in str(note) for note in ode.notes)

    # A row for people with no value (None), and a product of values by different dimensions
    ppl = ss.People(10_000, extra_states=[ss.FloatArr('since_birth', default=ss.choice([np.nan, 1.0, 10.0], p=[0.5, 0.25, 0.25]))])
    rate = ss.by(dict(since_birth=[None, 0, 5], value=[ss.peryear(0.0), ss.peryear(0.5), ss.peryear(0.1)]))
    sim = make(rate, people=ppl, stop=1).run()
    since, dead = sim.people.since_birth.raw[sim.people.auids], sim.diseases.d.D.raw[sim.people.auids]
    assert dead[np.isnan(since)].sum() == 0 and dead[since == 1].mean() > 2*dead[since == 10].mean()
    both = ss.by(sex=dict(female=1.0, male=2.0))*ss.by(urban={False:1.0, True:3.0})
    assert both.dims == ['sex', 'urban']

    # A distribution for each level of a property, as a dwell time
    A, B = ss.states('A B')
    dur = ss.by(sex=dict(female=ss.lognorm_ex(ss.months(6), ss.months(2)), male=ss.lognorm_ex(ss.months(12), ss.months(4))))
    sim = ss.Sim(ss.Disease(A, B, go=ss.progression('A -> B', dur=dur), name='d'), n_agents=10_000, unit='month', stop=3, verbose=0).init()
    uids = sim.people.auids
    drawn = sim.diseases.d.engine.sample_dwell(sim.diseases.d.graph.transitions['go'], uids) # In months
    female = sim.people.female.raw[uids]
    assert np.isclose(drawn[female].mean(), 6, rtol=0.05) and np.isclose(drawn[~female].mean(), 12, rtol=0.05)
    return sim


@sc.timer()
def test_many_groups():
    sc.heading('Testing more groups than 16-bit integers hold, as agents (TL6)')
    S, D = ss.states(2)
    dis = ss.Disease(onset=S >> D @ ss.peryear(1), back=D >> S @ ss.days(7), name='dis')
    dis.stratify(a=[f'a{i}' for i in range(128)], states=[D])
    dis.stratify(b=[f'b{i}' for i in range(384)], states=[D]) # 49,152 groups: an OverflowError, or groups wrapped round to negative numbers
    sim = ss.Sim(diseases=dis, n_agents=2_000, start='2010-01-01', stop='2010-02-01', dt=ss.days(1), verbose=0).run()
    groups = sim.groups
    assert groups.n_groups == 49_152
    assert np.array_equal(groups.group_raw(), groups.level_raw(0).astype(np.int64)*384 + groups.level_raw(1)) # Kept up to date as people move in and out of D
    res = sim.results.dis
    assert res.n_D.by('b').values[-1].sum() == res.n_D[-1] == sim.diseases.dis.D.sum()
    return sim


@sc.timer()
def test_given_splits():
    sc.heading('Testing splits given by the people and over several dimensions (TL3, TL4)')
    # Stratifying by a property the people already have keeps it, and is not described as guessed (TL3: everyone was put at False, the first level)
    N = 10_000
    tob = np.random.default_rng(1).random(N) < 0.3
    ppl = ss.People(N, extra_states=[ss.BoolState('tob', default=lambda k: tob[:k] if k == N else np.zeros(k, bool))])
    A, B = ss.states('A B')
    d = ss.Disease(A, B, go=A >> B @ ss.peryear(0.1)*ss.by(tob=[0.0, 1.0]), name='d') # Only smokers move
    sim = ss.Sim(d, people=ppl, stratify=dict(tob=[False, True]), start=2000, stop=2005, dt=1, verbose=0).run()
    assert sim.people.tob.sum() == tob.sum()
    expected = tob.sum()*(1 - np.exp(-0.5))
    assert abs(sim.results.d.go.sum() - expected) < 4*np.sqrt(expected)
    assert 'guessed' not in [line for line in sim.describe(output=True).splitlines() if 'tob' in line and 'dimension' in line][0]

    # A split over two dimensions that contradicts one's own split raises (TL4: urban came out at 0.497 where 0.17 was given)
    table = np.array([[0.11, 0.21, 0.22, 0.23, 0.23], [0.75, 0.16, 0.05, 0.02, 0.02]]) # P(wealth | urban)
    def make(table):
        m = ss.Disease(stay=ss.progression('S -> S', rate=ss.peryear(0)), name='life')
        m.stratify(urban=[False, True], split=[0.83, 0.17])
        m.stratify(wealth=['w1', 'w2', 'w3', 'w4', 'w5'], split=ss.by(table, dims=['urban', 'wealth']))
        return ss.Sim(m, n_agents=20_000, start=2010, stop=2011, verbose=0)
    with pytest.raises(ss.InconsistentDimension, match='0.83'):
        make(table).init()
    n = make(table*np.array([0.83, 0.17])[:, None]).run().results.life.n_S # As a joint distribution, it agrees
    assert np.isclose(n.select(urban=1).values[0]/n.values[0], 0.17, atol=0.01)
    assert np.isclose(n.select(wealth=0).values[0]/n.values[0], 0.83*0.11 + 0.17*0.75, atol=0.01)
    return sim


@sc.timer()
def test_counters():
    sc.heading('Testing counters: increment= on transitions, deliveries, and transfers')
    def make(solver, twins=0.0, **kw):
        NP, P = ss.states('NP P')
        conceive = ss.Transition(NP, P, rate=ss.by(age=[ss.peryear(0), ss.peryear(0.25), ss.peryear(0.15), ss.peryear(0.05)]))
        single = ss.Transition(P, NP, dur=ss.months(9), p=1 - twins, increment='parity')
        double = ss.Transition(P, NP, dur=ss.months(9), p=twins, increment=dict(parity=2)) # Twins add 2
        sim = ss.Sim(ss.Disease(NP, P, conceive=conceive, single=single, double=double, name='preg'), solver=solver, start=2000, stop=2010, dt=ss.months(1), use_aging=False, verbose=0, **kw)
        sim.stratify(age=[0, 15, 25, 35])
        sim.stratify(parity=list(range(8)))
        return sim.run()
    def mean_parity(sim): # By age group, of those 15 and over
        res = sim.results.preg
        n = [res.n_NP.select(age=a).by('parity').values[-1] + res.n_P.select(age=a).by('parity').values[-1] for a in range(1, 4)]
        return np.array([(k*np.arange(8)).sum()/k.sum() for k in n])

    # Mean parity by age, agents against the ODE (0.7% apart at most, measured), and twins add 2
    ode, abm = mean_parity(make('ode', total_pop=100_000)), mean_parity(make('abm', n_agents=100_000))
    assert np.allclose(abm, ode, rtol=0.01) and ode[0] > 1.5
    sim = make('abm', twins=1.0, n_agents=2000)
    parity = sim.people.parity[sim.people.auids]
    assert np.all(parity % 2 == 0) and parity.max() > 7 # The last level is open-ended, and agents keep counting
    assert np.allclose(mean_parity(make('ode', twins=0.5, total_pop=5000)), 1.5*ode, rtol=0.02) # A little under, since the ODE's 7+ is 7

    # A delivery adds a dose to the people it reaches, a transfer takes a level of immunity off at a rate, and recovery counts episodes
    def campaign(solver, **kw):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(S, I, R, infection=S >> I @ 0.2*I, recovery=ss.progression('I -> R', dur=ss.days(10), increment='episodes'), init=dict(I=0.01), name='sir')
        sir.stratify(waning=ss.transfer(increment=dict(immunity=-1), rate=ss.peryear(0.5)), immunity=[0, 1, 2, 3], split=[0, 0, 0, 1])
        camp = ss.Delivery(increment='doses', coverage=0.6, at=[1, 2, 3], name='camp')
        sim = ss.Sim(sir, interventions=camp, solver=solver, start=0, stop=5, dt=ss.days(1), unit='year', verbose=0, **kw)
        sim.stratify(doses=[0, 1, 2, 3], episodes=[0, 1, 2])
        return sim.run()
    def total(sim, dim):
        res = sim.results.sir
        return res.n_S.by(dim).values[-1] + res.n_I.by(dim).values[-1] + res.n_R.by(dim).values[-1]
    ode, abm = campaign('ode', total_pop=20_000), campaign('abm', n_agents=20_000)
    exact = 20_000*np.array([0.4**3, 3*0.6*0.4**2, 3*0.6**2*0.4, 0.6**3]) # Doses after three rounds at 60%: binomial
    assert np.allclose(total(ode, 'doses'), exact) and np.allclose(total(abm, 'doses'), exact, rtol=0.05)
    m = 2.5 # Levels lost in 5 years at 0.5 per year: Poisson
    lost = np.array([1 - sum(np.exp(-m)*m**k/math.factorial(k) for k in range(3))] + [np.exp(-m)*m**k/math.factorial(k) for k in [2, 1, 0]])
    assert np.allclose(total(ode, 'immunity'), 20_000*lost, rtol=1e-3) and np.allclose(total(abm, 'immunity'), 20_000*lost, rtol=0.05)
    assert np.isclose(total(ode, 'episodes')[1], ode.results.sir.recovery.values.sum())

    # Written to the spec and back as code
    spec = abm.to_spec()
    code = ss.to_code(spec)
    assert "increment='episodes'" in code and 'increment=dict(immunity=-1)' in code and "increment='doses'" in code
    ns = {}
    exec(code.replace("if __name__ == '__main__':", 'if False:'), ns)
    assert ns['sim'].hash() == abm.hash()
    with pytest.raises(ValueError, match='whole number'):
        ss.progression('I -> R', dur=10, increment=dict(episodes=0.5))
    with pytest.raises(ss.MissingDimension, match='stratify'):
        ss.Sim(ss.Disease(ss.progression('I -> R', dur=10, increment='episodes'), init=dict(I=0.1), name='d'), n_agents=100, verbose=0).run()
    return abm


@sc.timer()
def test_clocks():
    sc.heading('Testing clocks: the time since an event')
    def make(solver, **kw):
        NP, P = ss.states('NP P')
        act = ss.par(ss.by(dict(since_birth=[None, 0, 1, 2, 3, 6, 12, 24], value=[1.0, 0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]))) # Activity by months since the last birth; None: no birth yet
        preg = ss.Disease(NP, P, conceive=NP >> P @ ss.peryear(0.5)*act, deliver=ss.Transition(P, NP, dur=ss.months(9), move=dict(since_birth=0)), act=act, name='preg')
        sim = ss.Sim(preg, solver=solver, start=2000, stop=2010, dt=ss.months(1), use_aging=False, verbose=0, **kw)
        sim.stratify(since_birth=ss.months([0, 1, 2, 3, 6, 12, 24]), kind='clock')
        return sim.run()
    def shares(sim): # The share in each level as given (a compartmental run has a level for each step)
        strat = sim.groups.strats[sim.groups.dim_index('since_birth')]
        res = sim.results.preg
        n = res.n_NP.by('since_birth').values[-1] + res.n_P.by('since_birth').values[-1]
        n = n if strat.coarse is None else np.bincount(strat.coarse, weights=n)
        return n/n.sum()
    ode = shares(make('ode', total_pop=100_000))
    assert make('ode', total_pop=1000).groups.strats[0].n == 26 # 0 to 24 months a month at a time, and none
    for solver,kw in [('binomial', dict(total_pop=100_000)), ('abm', dict(n_agents=100_000))]:
        assert np.allclose(shares(make(solver, **kw)), ode, atol=0.01), f'Expected {solver} to match the ODE'
    sim = make('abm', n_agents=2000)
    since = sim.people.since_birth[sim.people.auids]
    born = sim.results.preg.deliver.values.sum() > 0
    assert born and np.isnan(since).any() and np.allclose(since[~np.isnan(since)] % 1, 0) # Whole months, and none for those with no birth
    assert np.isclose(shares(sim)[-1], np.isnan(since).mean()) # none holds those with no birth
    spec = sim.to_spec()
    code = ss.to_code(spec)
    assert "since_birth=ss.months([0.0, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0])" in code and "kind='clock'" in code and 'move=dict(since_birth=0.0)' in code
    ns = {}
    exec(code.replace("if __name__ == '__main__':", 'if False:'), ns)
    assert ns['sim'].hash() == sim.hash()
    with pytest.raises(ss.BadSplit, match='none'):
        bad = ss.Sim(ss.Disease(ss.progression('I -> R', dur=10), init=dict(I=0.1), name='d'), n_agents=100, verbose=0)
        bad.stratify(since=[0, 1], kind='clock', split=[0.5, 0.5, 0])
        bad.init()
    return sim


@sc.timer()
def test_rows():
    sc.heading('Testing p= with a to axis (a table over the destinations), and a function giving a row')
    from scipy.special import expit
    methods = ['pill', 'iud', 'inj']
    def make(solver, fn=False, **kw):
        none = ss.State('none')
        on = {m: ss.State(m) for m in methods}
        b0 = ss.par(-0.5)
        p_use = expit(b0 + 1.0*ss.ppl.urban)
        mix = ss.par(ss.by(dict(age=[0, 0, 0, 30, 30, 30], to=methods*2, value=[0.6, 0.3, 0.1, 0.2, 0.5, 0.3]))) # The switching row by age
        w = ss.par(ss.by(to=[1.0, 2.0, 1.0])) # Weights by method, by position
        row = lambda sim, uids: np.tile([0.5, 0.2, 0.2, 0.1], (len(uids), 1)) # A function giving a row for each person
        trs = {f'from_{src.name}':ss.Transition(src, [none, *on.values()], dur=ss.months(12), p=row if fn else [1 - p_use, p_use*mix*w]) for src in [none, *on.values()]}
        sim = ss.Sim(ss.Disease(none, *on.values(), **trs, b0=b0, mix=mix, w=w, name='contra'), solver=solver, start=2000, stop=2020, dt=ss.months(1), use_aging=False, verbose=0, **kw)
        sim.stratify(age=[0, 30])
        sim.stratify(urban=[False, True], split=[0.4, 0.6])
        return sim
    ode = make('ode', total_pop=20_000).run()
    abm = make('abm', n_agents=20_000).run()
    n = lambda sim: np.array([sim.results.contra[f'n_{m}'][-1] for m in ['none'] + methods])
    young = ode.groups.strats[0].split[0] # Everyone re-decides each year from the same row, so the shares are those of the row
    use = 0.4*expit(-0.5) + 0.6*expit(0.5)
    expected = 20_000*np.array([1 - use] + list(use*(young*np.array([0.6, 0.6, 0.1])/1.3 + (1 - young)*np.array([0.2, 1.0, 0.3])/1.5)))
    assert np.allclose(n(ode), expected, rtol=0.01) and np.allclose(n(abm), expected, rtol=0.05)
    assert np.allclose(n(make('abm', fn=True, n_agents=10_000).run()), 10_000*np.array([0.5, 0.2, 0.2, 0.1]), rtol=0.06)
    tr = abm.diseases.contra.graph.transitions.from_none
    assert len(tr.p) == 4 and len(tr.p_given) == 2
    assert 'mix * w normalized over pill, iud, inj' in '\n'.join(abm.diseases.contra.graph.describe(abm))
    spec = abm.to_spec()
    assert spec['modules'][0]['transitions'][0]['p'][1] == 'expit(b0 + (1.0 * people.urban)) * mix * w' # As given
    code = ss.to_code(spec)
    ns = {}
    exec(code.replace("if __name__ == '__main__':", 'if False:'), ns)
    assert ns['sim'].hash() == abm.hash()
    with pytest.raises(ss.UnknownState, match='implant'):
        none, pill = ss.states('none pill')
        ss.Disease(go=ss.Transition(none, [none, pill], rate=ss.peryear(1), p=[0.5, 0.5*ss.par(ss.by(to=dict(pill=1.0, implant=1.0)))]), name='bad')
    return abm


def test_contacts_dims():
    sc.heading('Testing one contact matrix over several stratifications together')
    sex, risk = np.array([[0.1, 1.0], [1.0, 0.1]]), np.array([[1.0, 0.5], [0.5, 3.0]])
    def run(net, solver='ode', seed=1):
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.1)), recovery=ss.progression('I -> R', dur=ss.days(10)), init_prev=0.01, name='sir')
        sim = ss.Sim(sir, networks=net, stratify=[dict(sex=['female', 'male']), dict(risk=['low', 'high'], split=[0.8, 0.2])], total_pop=20_000, n_agents=20_000, solver=solver, unit='day', stop=120, rand_seed=seed, **kw)
        return sim.run().results.sir.n_R.values

    # A separable matrix given whole is the product of one for each (ST23)
    for solver in ['ode', 'abm']:
        assert np.allclose(run(ss.contacts(sex=sex, risk=risk), solver), run(ss.contacts(np.kron(sex, risk), dims=['sex', 'risk']), solver))

    # One that is not: men mix with women, and the high-risk with each other more among men; as (2, 2, 2, 2) or (4, 4), first stratification slowest
    m = np.zeros((2, 2, 2, 2))
    m[0, :, 1, :] = [[0.8, 0.2], [0.5, 1.5]] # Women's contacts with men
    m[1, :, 0, :] = [[0.8, 0.5], [0.2, 2.0]] # Men's contacts with women
    ode = run(ss.contacts(m, dims=['sex', 'risk']))
    assert np.allclose(ode, run(ss.contacts(m.reshape(4, 4), dims=['sex', 'risk'])))
    abm = np.mean([run(ss.contacts(m, dims=['sex', 'risk']), 'abm', seed) for seed in range(3)], axis=0)
    assert np.isclose(abm[-1], ode[-1], rtol=0.1), (abm[-1], ode[-1]) # As with any contact matrix (about 7% apart for the product form here too)
    with pytest.raises(ss.ShapeError, match='risk'):
        run(ss.contacts(np.ones((6, 6)), dims=['sex', 'risk']))
    return

if __name__ == '__main__':
    mixing  = test_mixing()
    states  = test_states()
    strains = test_strains()
    carriage = test_carriage()
    space   = test_space()
    register = test_register()
    ragged  = test_ragged()
    sex     = test_sex()
    values  = test_property_values()
    props   = test_properties_in_rates()
    by_props = test_by_properties()
    many    = test_many_groups()
    splits  = test_given_splits()
    counters = test_counters()
    clocks  = test_clocks()
    rows    = test_rows()


    cdims = test_contacts_dims()
