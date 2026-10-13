"""
Test models written as declared states and transitions, and the solvers that run them
"""

# %% Imports and settings
import pytest
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(verbose=0)


def make_sir(**kwargs):
    """ The two-line SIR """
    sir = ss.Disease(
        infection = ss.transmission('S -> I', beta=0.3),
        recovery  = ss.progression('I -> R', dur=10),
        init_prev = 0.01,
        name = 'sir',
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

    sim = ss.Sim(sir, stop=10, **kw).init()
    assert 'sir' in sim.diseases # The module is named from its variable
    assert sim.unit == 'day' # From the typed durations

    # A state with no name is an error
    with pytest.raises(ss.CodedError, match='E210'):
        states = ss.states(2)
        ss.Disease(states[0] >> states[1] @ 0.1)

    # A misspelled state is an error if states were declared
    with pytest.raises(ss.CodedError, match='E204'):
        S, I = ss.states(2)
        ss.Disease(ss.Transition(S, I, rate=0.1 * I), ss.progression('I -> Q', dur=3))

    # Reserved names, undriven transitions, and bad splits
    with pytest.raises(ss.CodedError, match='E201'):
        ss.Disease(step=ss.progression('I -> R', dur=3))
    with pytest.raises(ss.CodedError, match='E209'):
        ss.Disease(recovery=ss.Transition('I', 'R'))
    with pytest.raises(ss.CodedError, match='E205'):
        ss.Disease(a=ss.progression('I -> H', p=0.2, dur=3), b=ss.progression('I -> R', p=0.7, dur=3))
    with pytest.raises(ss.CodedError, match='E208'):
        S, I = ss.states(2)
        ss.Disease(infection = S >> I @ 0.1 * I * I)
    return sir


@sc.timer()
def test_time():
    sc.heading('Testing how the sim gets its unit')
    sim = ss.Sim(make_sir(), stop=100, **kw).init()
    assert sim.unit is None # Nothing has a unit, so time is in timesteps
    assert sim.t.npts == 101
    assert ss.Sim(make_sir(), stop=100, unit='day', **kw).init().unit == 'day'
    assert ss.Sim(make_sir(), stop=10, dt=ss.weeks(1), **kw).init().unit == 'week'
    assert ss.Sim(make_sir(), start='2020-01-01', stop='2020-03-01', **kw).init().unit == 'day'
    assert ss.Sim(make_sir(), start=2020, stop=2030, **kw).init().unit == 'year'

    # A rate and its duration say the same thing, whatever the unit
    a = ss.Disease(recovery=ss.progression('I -> R', dur=ss.weeks(1)), init=dict(I=1.0))
    b = ss.Disease(recovery=ss.progression('I -> R', rate=ss.perday(1/7)), init=dict(I=1.0))
    res = [ss.Sim(mod, stop=30, unit='day', **kw).run().diseases[0].results.n_I for mod in [a, b]]
    assert np.allclose(res[0], res[1])
    assert np.isclose(res[0][7], 10_000*np.exp(-1), rtol=1e-4)
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
            infection = ss.transmission('S -> I', beta=0.3),
            severe    = ss.progression('I -> H', p=0.1, dur=5),
            mild      = ss.progression('I -> R', p=0.9, dur=10),
            discharge = ss.progression('H -> R | D', dur=7, p=[0.8, 0.2]),
            dying     = ss.death('I ->', rate=0.005),
            init_prev = 0.01,
        ),
        cholera = make_cholera(),
    )
    sims = sc.objdict()
    for name,mod in models.items():
        ref = None
        for solver in ['ode', 'binomial', 'tau', 'sde', 'abm']:
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
            assert np.isclose(peak, ref, rtol=0.07), f'Expected {solver} to match the ODE within 7% for {name}, but {peak:n} vs {ref:n}'
            sims[f'{name}_{solver}'] = sim

    # The exact stochastic solver, at a smaller size, and the one chosen automatically
    sim = ss.Sim(make_sir(), total_pop=2000, stop=100, solver='gillespie', **kw).run()
    assert sim.results.sir.n_R[-1] > 1000
    assert ss.Sim(make_sir(), total_pop=2000, solver='stochastic', **kw).init().solver == 'gillespie'
    assert ss.Sim(make_sir(), total_pop=1e8, solver='stochastic', **kw).init().solver == 'tau'
    assert ss.Sim(make_sir(), solver='ctmc', **kw).init().solver == 'gillespie'

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
    assert ss.Sim(make_sir(), stop=10, n_agents=1000, **kw).init().solver == 'abm'
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
    with pytest.raises(ss.CodedError, match='E804'):
        ss.Sim(sir, stop=100, approx=False, solver='ode', **kw).init()

    # A module written as code cannot run as compartments
    with pytest.raises(ss.CodedError, match='E804'):
        ss.Sim(make_sir(), demographics=ss.Births(), solver='ode', start=2000, stop=2010, **kw).init()
    return sim


@sc.timer()
def test_counts():
    sc.heading('Testing states that are not people')
    cholera = make_cholera()
    graph = cholera.graph
    assert [s.name for s in graph.count_states] == ['W'] # No transition connects W to S
    assert 'states.W.people' in graph.guessed
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

    # A bare count in a rate is an error
    with pytest.raises(ss.CodedError, match='E211'):
        S, I, W = ss.states(3)
        ss.Disease(infection = S >> I @ 0.5 * W, shedding = ss.null >> W @ 10 * I.n)
    return sim


@sc.timer()
def test_describe():
    sc.heading('Testing describe(), equations(), and the transition table')
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

    eqs = sir.equations(output=True)
    print(eqs)
    assert eqs.splitlines() == ['dS/dt = -beta * S * I / N', 'dI/dt =  beta * S * I / N - I / dur', 'dR/dt =  I / dur']
    table = repr(sir.transitions)
    print(table)
    assert 'contact-mediated, source I' in table
    print(sim.solvers(output=True))
    spec = sim.to_spec()
    assert spec['modules'][0]['transitions'][0]['rate'] == 'beta * I'
    return sim


@sc.timer()
def test_stratify():
    sc.heading('Testing stratification by age with a contact matrix')
    matrix = [[12, 4, 1], [3, 8, 2], [1, 3, 4]] # Contacts per day of each age group (rows) with each age group
    sims = sc.objdict()
    for solver in ['ode', 'binomial', 'abm']:
        sir = make_sir().stratify(age=[0, 15, 65], adjust={'recovery': [1.5, 1.0, 0.5]})
        sir.pars.beta.value = 0.03
        pars = dict(n_agents=20_000) if solver == 'abm' else {}
        sim = ss.Sim(sir, networks=ss.contacts(matrix, age=[0, 15, 65]), total_pop=100_000, unit='day', stop=150, solver=solver, **pars, **kw).run()
        sims[solver] = sim
    res = sims.ode.results.sir
    by_age = res.n_S.by('age')
    assert by_age.values.shape == (151, 3)
    assert np.allclose(by_age.values.sum(axis=1), res.n_S) # The strata sum to the total
    assert np.allclose(res.n_I.select(age=0), res.n_I.by('age').values[:, 0]) # By position
    assert np.allclose(res.n_I.select(age='65+'), res.n_I.by('age').values[:, 2]) # By label
    assert np.allclose(by_age.values[0], [14_850, 49_500, 34_650]) # The split follows the width of the age groups, less the 1% infected
    attack = [res.infection.select(age=i).sum()/by_age.values[0, i] for i in range(3)]
    assert attack[0] > attack[2], 'Children have the most contacts, so should have the highest attack rate'
    for solver in ['binomial', 'abm']:
        other = sims[solver].results.sir
        assert np.isclose(other.cum_infection[-1], res.cum_infection[-1], rtol=0.05), f'Expected {solver} to match the ODE'
        assert np.allclose(other.infection.by('age').values.sum(axis=0), res.infection.by('age').values.sum(axis=0), rtol=0.1)

    # A mixing pool of everyone and a random network with the same number of contacts agree, as agents and as an ODE
    finals = sc.objdict()
    for key,pars in dict(pool_ode=dict(networks=ss.MixingPool(n_contacts=10)), pool_abm=dict(networks=ss.MixingPool(n_contacts=10), n_agents=20_000),
                         net_abm=dict(networks=ss.RandomNet(n_contacts=10), n_agents=20_000), net_ode=dict(networks=ss.RandomNet(n_contacts=10), solver='ode')).items():
        sir = make_sir()
        sir.pars.beta.value = 0.03
        sim = ss.Sim(sir, total_pop=100_000, unit='day', stop=150, **pars, **kw).run()
        finals[key] = sim.results.sir.cum_infection[-1]
        if key == 'net_ode':
            assert any('approximated' in note for note in sim.notes) # Asking for an ODE of a network model gives the standard approximation, and says so
    assert np.allclose(finals[:], finals[0], rtol=0.03), f'Expected the same attack rate from each, not {finals}'

    # Stratifying the sim gives every disease the same groups
    def make(name, beta, dur):
        return ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> S', dur=dur), init_prev=0.01, name=name)
    pars = dict(unit='day', stop=100, **kw)
    both = ss.Sim([make('flu', 0.4, 5), make('rsv', 0.3, 8)], **pars).stratify(age=[0, 5, 18, 65]).run()
    same = ss.Sim([make('flu', 0.4, 5), make('rsv', 0.3, 8)], stratify=dict(age=[0, 5, 18, 65]), **pars).init()
    assert both.hash().model == same.hash().model
    for key in ['flu', 'rsv']:
        assert both.results[key].n_I.by('age').values.shape == (101, 4)

    # Errors
    with pytest.raises(ss.CodedError, match='E305'):
        make_sir().stratify(age=[0, 15]).stratify(age=[0, 15])
    with pytest.raises(ss.CodedError, match='E301'):
        make_sir().stratify(age=[0, 15], adjust={'recovry': [1, 2]})
    with pytest.raises(ss.CodedError, match='E302'):
        make_sir().stratify(risk=['low', 'high'], split=[0.5, 0.3, 0.2])
    with pytest.raises(ss.CodedError, match='E308'):
        a = make_sir().stratify(age=[0, 15])
        b = ss.Disease(recovery=ss.progression('I -> R', dur=3), name='other').stratify(age=[0, 20])
        ss.Sim([a, b], **kw).init()
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
    assert ode.cum_vaccination[-1] > 50_000
    assert ode.infection.select(risk='high').sum()/20_000 > ode.infection.select(risk='low').sum()/80_000 # The high-risk group has the higher attack rate
    assert ode.infection.select(vaccinated=True).sum() < 0.1*ode.infection.sum() # The disease never mentions vaccination
    for key in ['cum_infection', 'cum_vaccination']:
        assert np.isclose(abm[key][-1], ode[key][-1], rtol=0.1), f'Expected agents to match the ODE for {key}, but {abm[key][-1]:n} vs {ode[key][-1]:n}'

    # The unvaccinated model has a larger epidemic
    plain = make_sir()
    plain.pars.beta.value = 0.2
    plain.stratify(risk=['low', 'high'], split=[0.8, 0.2], adjust={'infection': [1.0, 3.0]})
    base = ss.Sim(plain, total_pop=100_000, unit='day', stop=150, **kw).run()
    assert base.results.sir.cum_infection[-1] > 1.2*ode.cum_infection[-1]

    # A transfer along a dimension the model does not have is an error
    with pytest.raises(ss.CodedError, match='E1102'):
        ss.Sim(ss.Disease(recovery=ss.progression('I -> R', dur=3), move=ss.transfer(risk='low -> high', rate=0.1)), **kw).init()
    return sims


@sc.timer()
def test_outcomes():
    sc.heading('Testing outcomes, data, and calibration')
    def make(beta):
        beta = ss.par(beta, bounds=[0.05, 1.0])
        S, I, R = ss.states(3)
        infection = S >> I @ beta * I
        recovery = I >> R @ ss.days(10)
        cases  = ss.outcome(0.5 * infection, delay=ss.days(5), every=ss.weeks(1))
        hosp   = ss.outcome(0.05 * infection, delay=ss.days(7), duration=ss.days(10))
        deaths = ss.outcome(0.2 * hosp, delay=ss.lognorm_ex(mean=14, std=5, unit='days'))
        return ss.Disease(infection, recovery, cases, hosp, deaths, prevalence=I/(S + I + R), init_prev=0.01, name='sir')

    pars = dict(total_pop=1e6, start='2020-01-01', stop='2020-06-01', **kw)
    truth = ss.Sim(make(0.3), **pars).run()
    res = truth.results.sir
    total = res.infection.sum()
    assert truth.unit == 'day'
    assert np.isclose(res.cases.sum(), 0.5*total, rtol=0.01) # Half of infections are reported
    assert np.count_nonzero(res.cases) <= len(res.cases)/7 + 1 # ... as weekly totals
    assert np.isclose(res.hosp.sum(), 0.05*total, rtol=0.01)
    assert np.isclose(res.deaths.sum(), 0.01*total, rtol=0.02) # Outcomes chain
    assert np.argmax(res.hosp) == np.argmax(res.infection) + 7 # The delay
    assert res.n_hosp.max() > res.hosp.max() # The number in hospital, as well as admissions
    assert np.isclose(res.prevalence.max(), res.n_I.max()/1e6)

    # Fit beta to the weekly cases
    df = sc.dataframe(date=truth.t.timevec[6::7], cases=res.cases.values[6::7])
    sim = ss.Sim(make(0.15), data=df, **pars)
    fit = sim.calibrate()
    fit.summarize()
    assert np.isclose(fit.pars.beta, 0.3, rtol=0.01), f'Expected to recover beta=0.3, not {fit.pars.beta}'
    assert fit.mismatch < 0.01 < fit.mismatch_before
    assert np.isclose(sim.new().init().diseases.sir.pars.beta.value, 0.15) # The original is not changed
    refit = ss.Sim(make(0.15), pars=fit.pars, **pars).run() # The best-fit values can be passed to another sim
    assert np.allclose(refit.results.sir.n_I, fit.sim.results.sir.n_I)

    # Mistakes in the data and the parameters are caught by name
    with pytest.raises(ss.CodedError, match='E705'):
        ss.Sim(make(0.3), data=df.rename(columns=dict(cases='casess')), **pars).init()
    renamed = ss.data(df.rename(columns=dict(cases='Reported')), aliases=dict(cases='Reported'))
    assert 'cases' in ss.Sim(make(0.3), data=renamed, **pars).init().data.columns
    with pytest.raises(ss.CodedError, match='E602'):
        ss.Sim(make(0.3), pars=dict(betta=0.2), **pars).init()
    with pytest.raises(ss.CodedError, match='E1302'):
        ss.Sim(make(0.3), **pars).calibrate()
    with pytest.raises(ss.CodedError, match='E702'):
        ss.Sim(ss.Disease(recovery=ss.progression('I -> R', dur=3), cases=ss.outcome('infection')), **kw).init()
    return fit


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
    assert msim.sims[0].hash().model == msim.sims[1].hash().model

    # A modifier can also be given as a path, and combines with others by multiplication
    both = [ss.modify('sir.pars.beta', multiply=0.5, start=20, name='a'), ss.modify('sir.pars.beta', multiply=0.5, start=20, name='b')]
    twice = ss.Sim(sir, interventions=both, **pars).run()
    assert twice.results.sir.n_R[-1] < n_R.select(scenario='masks').values[0, -1]
    with pytest.raises(ss.CodedError, match='E504'):
        ss.Sim(sir, interventions=ss.modify('sir.pars.betta', multiply=0.5), **pars).init()

    # One sim is one run; many runs are a MultiSim
    sim = ss.Sim(sir, solver='binomial', total_pop=1000, stop=60, **kw)
    runs = ss.parallel(sim, n_runs=8)
    n_I = runs.results.sir.n_I
    assert n_I.values.shape == (8, 61)
    assert len(np.unique(n_I.values[:, 30])) > 1 # Each run has its own seed
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
            infection = ss.transmission('S -> I', beta=0.3),
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
    by_age = ss.Sim(make(), stop=100, unit='day', stratify=dict(age=[0, 50]), **kw).run()
    assert by_age.solver == 'ode'
    res = by_age.results.d
    assert res.n_H.select(age='50+').max() > res.n_H.select(age='0-49').max() # Older people are more likely to be hospitalized
    with pytest.raises(ss.CodedError, match='E804'):
        ss.Sim(make(), stop=100, unit='day', solver='ode', **kw).run()

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
    with pytest.raises(ss.CodedError, match='E403'):
        other = ss.Disease(infection=ss.transmission('S -> I', beta=dict(needles=0.1)), init_prev=0.05, name='hiv')
        ss.Sim(other, networks=ss.MFNet(), start=2000, stop=2005, **kw).run()

    # An Erlang dwell time is exact in every solver
    erl = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=ss.erlang(k=3, mean=ss.days(10))), init_prev=0.01, name='erl')
    stg = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=ss.days(10), stages=3), init_prev=0.01, name='stg')
    r1 = ss.Sim(erl, stop=100, **kw).run()
    r2 = ss.Sim(stg, stop=100, **kw).run()
    assert not r1.notes # No approximation
    assert np.allclose(r1.results.erl.n_I, r2.results.stg.n_I)
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

    # A seasonal value peaks once a year, under every solver
    for solver in ['ode', 'binomial', 'tau', 'abm']:
        beta = ss.seasonal(mean=ss.perday(0.3), amplitude=0.5, peak='January')
        sim = ss.Sim(make(beta), start=2020, stop=2023, dt=ss.days(1), solver=solver, total_pop=20_000, **kw).run()
        n_I = sim.results.sis.n_I[366:731] # The second year
        assert np.argmax(n_I) < 45 or np.argmax(n_I) > 320, f'Expected infections to peak in the winter with {solver}'
        assert n_I.max() > 1.5*n_I.min()
    assert sim.unit == 'day'
    assert 'seasonal(mean=' in sim.describe(output=True)

    # A fast transition is noted when the solver takes whole steps, and check_dt() says how much the step matters
    fast = ss.Disease(infection=ss.transmission('S -> I', beta=0.6), recovery=ss.progression('I -> R', dur=4), init_prev=0.01, name='fast')
    stepped = ss.Sim(fast, solver='binomial', unit='day', stop=60, total_pop=1e6, **kw)
    df = stepped.check_dt(output=True)
    assert any('rate × dt = 0.25' in note for note in stepped.new().init().notes)
    assert 0.05 < df.change[df.result == 'n_I'].iloc[0] < 0.2
    smooth = ss.Sim(fast, solver='ode', unit='day', stop=60, total_pop=1e6, **kw)
    assert smooth.check_dt(output=True).change.max() < 1e-3 # An ODE does not depend on the timestep
    assert not smooth.new().init().notes
    return sim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    forms = test_forms()
    names = test_names()
    time  = test_time()
    sims  = test_solvers(do_plot=do_plot)
    auto  = test_auto()
    count = test_counts()
    desc  = test_describe()
    strat = test_stratify()
    trans = test_transfer()
    fit   = test_outcomes()
    msim  = test_modify()
    funcs = test_functions()
    times = test_time_values()
