"""
Test dimensions that are more than a property of people: states with dimensions of their own, strains, and places
"""

# %% Imports and settings
import math
import pytest
import numpy as np
import pandas as pd
import scipy.integrate as sp_integrate
import scipy.stats as stats
import sciris as sc
import starsim as ss
import corpus

kw = dict(verbose=0)
solvers = ['ode', 'binomial', 'abm']


def make_towns(**kwargs):
    """ An SIR model in three towns, with nobody infected """
    towns = ss.geo.geography(names=['north', 'south', 'east'], populations=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
    S, I, R = ss.states('S I R')
    sir = ss.Disease(
        infection = ss.Transition(S, I, rate=0.3 * I),
        recovery  = ss.Transition(I, R, dur=10),
        init_prev = 0, # Seeded by seed_north()
        name = 'sir',
    )
    return sir.stratify(patch=ss.levels(towns, **kwargs))


def seed_north(sim):
    """ Infect 50 people in the first town, on the first step """
    if sim.ti == 0:
        sir = sim.diseases.sir
        if isinstance(sim.solver, ss.solvers.abm):
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
    sims = sc.obj()
    for solver in solvers:
        malaria = corpus.make_malaria(mortality=ss.days(10)*ss.by(species=[1.0, 0.5])) # Funestus dies twice as fast
        states = malaria.states
        malaria.stratify(species=ss.levels({'gambiae': 0.75, 'funestus': 0.25}, states=[states.Sv, states.Ev, states.Iv]))
        sims[solver] = corpus.run_with(malaria, solver, unit='day', stop=365, total_pop=20_000, stratify=dict(age=[0, 5, 15]))
    res = sims.ode.results.malaria
    assert [s.name for s in malaria.graph.count_states] == ['Sv', 'Ev', 'Iv'] # Nobody moves between humans and mosquitoes, so the mosquitoes are counts
    assert np.allclose(res.Sv.by('species').values[0], [37_500, 12_500], rtol=0.1) # Split by species (after the first day, in which funestus dies twice as fast) ...
    assert np.allclose(res.emergence.by('species').values[5], [3750, 1250])
    assert np.allclose(sum(res[f'{s}h'] for s in 'SEIR'), 20_000) # ... and left out of the population
    n_Iv = res.Iv.by('species').values[-1]
    assert n_Iv[1] < n_Iv[0]/6, 'Expected fewer of the species that is rarer and dies faster'
    assert res.Ih.by('age').values.shape[1] == 3
    for solver in ['binomial', 'abm']:
        other = sims[solver].results.malaria
        assert np.allclose(other.Iv.by('species').values[-1], n_Iv, rtol=0.2), f'Expected {solver} to match the ODE'
        assert np.isclose(other.Ih[-1], res.Ih[-1], rtol=0.1)

    return sims


@sc.timer()
def test_strains():
    sc.heading('Testing a strain: a new infection takes the strain of its source')
    sims = sc.obj()
    for solver in solvers:
        sims[solver] = corpus.run_with(corpus.make_tb(), solver, start=2000, stop=2050, dt=ss.years(0.1), total_pop=100_000)
    tb = sims.ode.diseases.tb
    res = tb.results
    assert tb.graph.strats[0].states == ['latent', 'active', 'on_treatment'] # A susceptible person has no strain
    assert getattr(res.susceptible, 'strata', None) is None
    by_strain = res.infection.cum.by('resistance').values[-1]
    assert res.active.by('resistance').values[0, 1] < 0.01 # Nobody starts with a resistant strain (so after the first step there is almost nobody)
    assert res.acquired.cum[-1] > 0 and by_strain[1] > 0 # It is acquired during treatment, and then transmitted
    assert by_strain[1] < 0.1*by_strain[0]
    for solver in ['binomial', 'abm']:
        other = sims[solver].results.tb
        assert np.isclose(other.infection.cum[-1], res.infection.cum[-1], rtol=0.05), f'Expected {solver} to match the ODE'
        assert np.isclose(other.active.by('resistance').values[-1, 0], res.active.by('resistance').values[-1, 0], rtol=0.1)
        assert 0 < other.infection.cum.by('resistance').values[-1, 1] < 0.1*by_strain[0]

    # Cross-immunity between strains: complete protection is the base model; none lets people in R be infected again with the other strain
    def flu(m=None, combine='max'):
        S, I, R = ss.states('S I R')
        strains = ss.Strains(['A', 'B'])
        if m is not None:
            strains.immunity = ss.immunity(due_to=R, protection=m, combine=combine)
        return ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=7), init_prev=0.01, name='flu', strains=strains)
    total = lambda *args, kwargs={}, **kws: ss.Sim(flu(*args, **kws), unit='day', stop=200, **kwargs, **kw).run().results.flu.infection.sum()
    once = total()
    assert np.isclose(total(np.ones((2, 2))), once)
    again = total(np.eye(2), kwargs=dict(solver='ode')) # Due to R, by the strain people carry there (the last), which a compartment keeps
    abm = total(np.eye(2), kwargs=dict(solver='abm', n_agents=10_000))
    assert again > 1.5*once and np.isclose(again, abm, rtol=0.1) # So A, B, A again, ...

    return sims


@sc.timer()
def test_carriage():
    sc.heading('Testing strains carried together: superinfection, and a transition that changes the strains carried')
    def make():
        S, L, I = ss.states(3)
        strains = ss.Strains(['A', 'B'], max_strains=2, immunity=ss.immunity(due_to=L, protection=0.5)) # Someone with A who is infected with B carries both, at half the rate
        A, B = strains
        return ss.Disease(
            infection  = S >> L @ ss.peryear(6)*I,
            activation = ss.Transition(L, I, rate=ss.peryear(0.5), who=~(A & B)),
            together   = ss.Transition(L, I, rate=ss.peryear(0.2), who=A & B), # Two strains progress together more slowly
            to_A       = ss.Transition(L, I, rate=ss.peryear(0.3), move=(A, B) >> A), # Or only one of the two progresses
            to_B       = ss.Transition(L, I, rate=ss.peryear(0.2), move=(A, B) >> B),
            recovery   = I >> S @ ss.peryear(1),
            init       = dict(I=0.02),
            name       = 'dis',
            strains    = strains,
        )

    # The reference: someone with A and B passes on each with half their infectiousness
    N = 100_000
    def deriv(t, y):
        S, LA, LB, LAB, IA, IB, IAB = y
        la, lb = 6*(IA + IAB/2)/N, 6*(IB + IAB/2)/N
        return [-(la + lb)*S + IA + IB + IAB, la*S - lb/2*LA - 0.5*LA, lb*S - la/2*LB - 0.5*LB, lb/2*LA + la/2*LB - 0.7*LAB,
                0.5*LA + 0.3*LAB - IA, 0.5*LB + 0.2*LAB - IB, 0.2*LAB - IAB]
    ref = sp_integrate.solve_ivp(deriv, [0, 20], [0.98*N, 0, 0, 0, 0.01*N, 0.01*N, 0], rtol=1e-10, atol=1e-6).y[:, -1]
    for solver in solvers:
        res = corpus.run_with(make(), solver, start=2000, stop=2020, dt=ss.years(0.02), total_pop=N).results.dis
        levels = lambda r: np.array([r.select(strain=label)[-1] for label in ['A', 'B', 'A+B']]) # Each set carried
        n_L, n_I = levels(res.L), levels(res.I)
        rtol = 1e-3 if solver == 'ode' else 0.15
        assert np.allclose(n_L, ref[1:4], rtol=rtol) and np.allclose(n_I, ref[4:], rtol=rtol), f'Expected {solver} to match the reference: {n_L}, {n_I} vs {ref[1:]}'
    return res


@sc.timer()
def test_strain_carriers():
    sc.heading('Testing where strains are carried: an infection, and what leads back to one')
    S, L, N, I, R = ss.states('S L N I R')
    foi = ss.peryear(8)*I
    def cleared(**kwargs): # R leads back to disease only by a new infection, so carries no strain
        return ss.Disease(S, L, I, R, infection=S >> L @ foi, reinfection=R >> L @ foi, activation=L >> I @ ss.peryear(1), clearance=L >> R @ ss.peryear(1),
                          recovery=I >> R @ ss.peryear(1), init=dict(I=0.01), name='d', strains=ss.Strains(['A', 'B'], max_strains=2, **kwargs))
    def staged(imm=None): # N is neither infectious nor entered by infection, but leads to I, so carries its strains
        return ss.Disease(S, L, N, I, R, infection=S >> L @ foi, progress=L >> N @ ss.peryear(1), worsen=N >> I @ ss.peryear(0.5), recovery=I >> R @ ss.peryear(1),
                          init=dict(I=0.01), name='d', strains=ss.Strains(['A', 'B'], max_strains=2, immunity=imm))
    res = corpus.run_with(cleared(), 'ode', start=2000, stop=2010, total_pop=20_000).results.d
    assert res.L.select(strain='A+B')[-1] == 0 and res.L[-1] > 0 # Reinfected from R with one strain, not added to one cleared
    plain = corpus.run_with(staged(), 'ode', start=2000, stop=2010, total_pop=1e5).results.d
    superinfected = corpus.run_with(staged(ss.immunity(due_to=N, protection=0.5)), 'ode', start=2000, stop=2010, total_pop=1e5).results.d
    assert np.isclose(superinfected.N[-1], plain.N[-1]) and superinfected.N.select(strain='A+B')[-1] > 0 # People in N carry another strain as well, and stay in N
    with pytest.raises(ValueError): # A transmission is worked out for the people it infects, who carry no strain yet: ss.Strains(..., infection=[...]) says it
        corpus.run_with(ss.Disease(S, I, infection=S >> I @ ss.by(strain=[1, 1.5])*ss.peryear(1)*I, init=dict(I=0.01), name='d', strains=ss.Strains(['A', 'B'])), 'ode', stop=2001)

    return res


@sc.timer()
def test_space():
    sc.heading('Testing places, with and without movement between them')
    mobility = sc.obj(
        none    = None,
        gravity = ss.geo.gravity(scale=2e-4), # Commuting, with the share of time away from the gravity model
        away    = ss.geo.gravity(away=0.2), # Commuting, with 20% of time spent away
        visit   = ss.geo.gravity(away=0.2, kind='visit'), # The same, but travelers do not infect the people they visit
        migrate = ss.geo.gravity(scale=2e-5, kind='migrate'),
        force   = ss.geo.gravity(away=0.2, kind='force'), # Nobody moves; 20% of each town's force of infection goes to the others
    )
    peaks = sc.obj()
    for key,mob in mobility.items():
        for solver in (solvers if key == 'gravity' else ['ode']): # Every kind as an ODE; gravity under each solver
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
            peaks[f'{key}_{solver}'] = res.I.by('patch').values.argmax(axis=0)
    assert peaks.gravity_ode[0] < peaks.gravity_ode[1] < peaks.gravity_ode[2] # The nearer town is reached first
    assert np.ptp(peaks.away_ode) < 3 < np.ptp(peaks.gravity_ode) # The more time people spend away, the closer together the towns peak

    # The passes over the agents are split between threads for a large population (here, for a small one): the results are those of one thread
    defaults = ss.solvers.numba.passes.min_threaded, ss.solvers.numba.passes.chunk_size
    ss.solvers.numba.passes.min_threaded, ss.solvers.numba.passes.chunk_size = 0, 1000
    try:
        sims = []
        for threads in [1, 0]: # One thread, then as many as Numba has
            with ss.options.context(threads=threads):
                sim = ss.Sim(make_towns(mobility=ss.geo.gravity(scale=2e-4)), stop=60, unit='day', n_agents=20_000, total_pop=100_000, **kw).init()
                sim.loop.insert(seed_north, label='sim.start_step')
                sims.append(sim.run())
    finally:
        ss.solvers.numba.passes.min_threaded, ss.solvers.numba.passes.chunk_size = defaults
    infections = [sim.results.sir.infection.values for sim in sims]
    assert infections[0].sum() > 0 and np.array_equal(*infections) # The same people infected, with threads or without

    # The radiation model has no parameters: of the people who travel from a place, the share who go to each other place comes from the populations of the places nearer than it
    line = ss.geo.geography(names=['a', 'b', 'c', 'd'], populations=[4e4, 1e4, 2e4, 3e4], x=[0, 10, 30, 70], y=[0, 0, 0, 0])
    M = ss.geo.radiation().matrix(ss.dimensions.Strat('patch', line))
    assert np.allclose(M[0], np.array([0, 4/20, 8/35, 12/70])/(1 - 4/10)) and np.allclose(M.sum(axis=1), 1) # m·n/((m + s)(m + n + s)), for the finite population (with no ties in distance, each row sums to 1)

    return peaks


@sc.timer()
def test_mixing():
    sc.heading('Testing contact matrices by dimension, and within places')
    def make():
        S, I, R = ss.states('S I R')
        sir = ss.Disease(
            infection = ss.Transition(S, I, rate=2 * I),
            recovery  = ss.Transition(I, R, rate=1/3),
            init_n    = dict(I=10),
            name      = 'sir',
        )
        sir.stratify(age=['young', 'old']) # Labels, so not the age that people have
        sir.stratify(location={'urban': 0.7, 'rural': 0.3})
        return sir
    mixing = ss.contacts(age=[[0.2, 0.3], [0.5, 0.7]], location=[[0.8, 0.2], [0.2, 0.8]])
    sims = sc.obj({solver:corpus.run_with(make(), solver, networks=mixing, total_pop=10_000, dt=0.1, stop=20) for solver in solvers})
    res = sims.ode.results.sir
    assert sims.ode.groups.strats[0].kind == 'plain'
    attack = {(a,l):res.R.select(age=a, location=l)[-1]/res.S.select(age=a, location=l)[0] for a in ['young', 'old'] for l in ['urban', 'rural']}
    assert attack['old', 'urban'] > attack['young', 'urban'] # Old people have more contacts
    assert np.isclose(attack['young', 'urban'], attack['young', 'rural']) # The two kinds of place mix alike and are seeded as their people are split, so they end alike
    for solver in ['binomial', 'abm']:
        assert np.isclose(sims[solver].results.sir.R[-1], res.R[-1], rtol=0.1), f'Expected {solver} to match the ODE'
    assert sims.abm.results.sir.I[0] <= 10 # As agents, the number who start in each state is exact (less anyone who has already recovered)

    # A split over two dimensions at once, a value for every combination of levels, and a contact matrix within each place
    def make_places(mobility=None):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.05) * I), recovery=ss.Transition(I, R, dur=ss.days(10)),
                         init_n=dict(I=ss.by(pd.DataFrame(dict(age=['0-4'], place=['a'], value=[10])))), name='sir')
        sir.stratify(age=[0, 5, 50])
        sir.stratify(place=ss.levels(['a', 'b'], fractions=ss.by([[3000, 1000], [5000, 2000], [2000, 500]], dims=['age', 'place']), kind='space', mobility=mobility))
        return ss.Sim(sir, networks=ss.contacts([[12, 4, 1], [4, 8, 2], [1, 2, 3]], ages=[0, 5, 50]), total_pop=13_500, unit='day', stop=100, **kw)
    start = corpus.at_start(make_places()).sir
    n = [sum(start[k].select(age=age, place=place)[0] for k in 'SIR') for age in ['0-4', '50+'] for place in ['a', 'b']]
    assert np.allclose(n, [3000, 1000, 2000, 500])
    assert start.I.select(age='0-4', place='a')[0] == 10 and start.I[0] == 10
    alone = make_places().run().results.sir
    assert alone.infection.cum.by('place').values[-1, 1] == 0 and alone.infection.cum[-1] > 5000 # Nobody moves, so the epidemic stays in a
    return sims


@sc.timer()
def test_by_properties():
    sc.heading('Testing values by a property of people that is not a dimension, and tables with bin edges')
    def make(mu, solver='auto', dims=None, **kw):
        S, D = ss.states('S D')
        sim = ss.Sim(ss.Disease(S, D, death=S >> D @ mu, name='d'), solver=solver, aging=False, verbose=0, **sc.mergedicts(dict(stop=10), kw))
        for dim,levels in (dims or {}).items():
            sim.stratify(**{dim:levels})
        return sim

    # A table with bin edges: each person's own age as agents; as compartments, only bins that each group is inside (bins that split a group need agents)
    table = ss.by(dict(age=[0, 20, 40, 60], value=[ss.peryear(v) for v in [0.01, 0.02, 0.03, 0.04]]))
    abm = make(table, n_agents=50_000, solver='abm').run()
    ode = make(table, total_pop=50_000, solver='ode', dims=dict(age=[0, 10, 20, 40, 60])).run() # Groups inside the bins: exact
    assert np.isclose(abm.results.d.D[-1], ode.results.d.D[-1], rtol=0.03)
    assert isinstance(make(table, total_pop=50_000, dims=dict(age=[0, 30, 60])).init().solver, ss.solvers.abm)

    # Several non-exponential dwell times out of a state, each for one sex: a track for each as compartments, as agents as ever
    def clearance(where_m=ss.person.male):
        S, A, R = ss.states('S A R')
        clear_f = ss.Transition(A, R, dur=ss.lognormal(ss.months(8), ss.months(2)), who=ss.person.female)
        clear_m = ss.Transition(A, S, dur=ss.lognormal(mean=ss.by(sex=dict(female=ss.months(6), male=ss.months(6))), std=ss.months(3)), who=where_m)
        return ss.Disease(S, A, R, clear_f=clear_f, clear_m=clear_m, init=dict(S=0.5, A=0.5), name='d').stratify(sex=['female', 'male'])
    left = sc.obj()
    for solver,pop in [('ode', dict(total_pop=20_000)), ('binomial', dict(total_pop=20_000)), ('abm', dict(n_agents=20_000))]:
        n = ss.Sim(clearance(), unit='month', dt=0.25, stop=10, solver=solver, verbose=0, **pop).run().results.d.A.by('sex').values
        left[solver] = n[32]/n[0] # Still infected at 8 months
    erlang = [stats.gamma.sf(8, 16, scale=8/16), stats.gamma.sf(8, 4, scale=6/4)] # 16 and 4 stages
    assert np.allclose(left.ode, erlang, atol=1e-3) and np.allclose(left.binomial, erlang, atol=0.03) and np.allclose(left.abm, [0.476, 0.208], atol=0.02) # Lognormal as agents (those who leave at 7.875 months or later, at the step nearest)
    sim = ss.Sim(clearance(where_m=ss.person.female), stop=10, verbose=0).init() # Both for women: none as compartments, so agents
    assert isinstance(sim.solver, ss.solvers.abm)

    # For the same people: the two clocks race, as a compartment for each pair of their Erlang stages, so the share leaving by each is exact
    def race(k=3):
        S, A, B, C = ss.states('S A B C')
        return ss.Disease(S, A, B, C, to_b=ss.Transition(A, B, dur=ss.gamma(k, scale=ss.months(10/k))), to_c=ss.Transition(A, C, dur=ss.gamma(2, scale=ss.months(7))), init=dict(A=1.0), default_state='S', name='d')
    exact = sp_integrate.quad(lambda t: stats.gamma.pdf(t, 3, scale=10/3)*stats.gamma.sf(t, 2, scale=7), 0, np.inf)[0] # P(the first clock ends first)
    r = ss.Sim(race(), unit='month', dt=0.25, stop=100, solver='ode', total_pop=1e6, verbose=0).run().results.d
    assert np.isclose(r.B[-1]/(r.B[-1] + r.C[-1]), exact, atol=1e-4)
    return sim


@sc.timer()
def test_counters():
    sc.heading('Testing counters: increment= on transitions, deliveries, and transfers')
    # A delivery adds a dose to the people it reaches, a transfer takes a level of immunity off at a rate, and recovery counts episodes
    def campaign(solver, **kw):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(S, I, R, infection=S >> I @ 0.2*I, recovery=ss.Transition(I, R, dur=ss.days(10), increment='episodes'), init=dict(I=0.01), name='sir')
        sir.stratify(waning=ss.transfer(increment=dict(immunity=-1), rate=ss.peryear(0.5)), immunity={0: 0, 1: 0, 2: 0, 3: 1})
        camp = ss.Event(increment='doses', when=[1, 2, 3], p=0.6, name='camp')
        sim = ss.Sim(sir, interventions=camp, solver=solver, start=0, stop=5, dt=ss.days(1), unit='year', verbose=0, **kw)
        sim.stratify(doses=[0, 1, 2, 3], episodes=[0, 1, 2])
        return sim.run()
    def total(sim, dim):
        res = sim.results.sir
        return res.S.by(dim).values[-1] + res.I.by(dim).values[-1] + res.R.by(dim).values[-1]
    ode, abm = campaign('ode', total_pop=20_000), campaign('abm', n_agents=20_000)
    exact = 20_000*np.array([0.4**3, 3*0.6*0.4**2, 3*0.6**2*0.4, 0.6**3]) # Doses after three rounds at 60%: binomial
    assert np.allclose(total(ode, 'doses'), exact) and np.allclose(total(abm, 'doses'), exact, rtol=0.05)
    m = 2.5 # Levels lost in 5 years at 0.5 per year: Poisson
    lost = np.array([1 - sum(np.exp(-m)*m**k/math.factorial(k) for k in range(3))] + [np.exp(-m)*m**k/math.factorial(k) for k in [2, 1, 0]])
    assert np.allclose(total(ode, 'immunity'), 20_000*lost, rtol=1e-3) and np.allclose(total(abm, 'immunity'), 20_000*lost, rtol=0.05)
    assert np.isclose(total(ode, 'episodes')[1], ode.results.sir.recovery.values.sum())
    return abm


if __name__ == '__main__':
    for name,test in list(globals().items()):
        if name.startswith('test_'):
            test()
