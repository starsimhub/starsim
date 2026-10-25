"""
Contract tests: properties every model and solver must have, checked on the model corpus
(see corpus.py). A failure here is usually a silent wrong answer somewhere else.
"""

# %% Imports and settings
import os
import runpy
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import corpus

stochastic = ['binomial', 'gillespie', 'abm']


# %% The guesses are marked

def guessed(text):
    """ The lines of describe() marked as guesses """
    return [line for line in text.splitlines() if '[guessed' in line or 'nothing in the model has a unit' in line]


def plain_sir(**kwargs):
    S, I, R = ss.states('S I R')
    return ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), name='sir', **kwargs)


def declared_sir(**kwargs):
    S, I, R = ss.states('S I R')
    return ss.Disease(S, I, R, infection=S >> I @ 0.3*I, recovery=I >> R @ ss.erlang(mean=10, k=3, unit='days'), init_prev=0.01, name='sir', **kwargs)


def lognormal(stages):
    dur = ss.lognorm_ex(mean=10, std=4, unit='days')
    S, I, R = ss.states('S I R')
    return ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=dur, stages=stages), init_prev=0.01, name='sir')


def fitted_sir(beta):
    S, I = ss.states('S I')
    return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), name='sir')


def unnamed(named):
    S, I, R = ss.states('S I R')
    infection, recovery = S >> I @ 0.3*I, I >> R @ ss.days(10)
    if named:
        return ss.Disease(infection, recovery, init_prev=0.01, name='sir')
    return ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init_prev=0.01, name='sir')


# Each row of the "what is guessed" table in 03-states-and-transitions.md: a sim that makes the guess, one that does not, and what describe() says
guesses = dict(
    dwell      = (lambda: ss.Sim(plain_sir()),                      lambda: ss.Sim(declared_sir()),                       'dwell ~'),
    stages     = (lambda: ss.Sim(lognormal(None), solver='ode'),     lambda: ss.Sim(lognormal(4), solver='ode'),           'stages'),
    unit       = (lambda: ss.Sim(plain_sir()),                      lambda: ss.Sim(plain_sir(), unit='day'),             'unit'),
    mixing     = (lambda: ss.Sim(plain_sir()),                      lambda: ss.Sim(plain_sir(), networks=ss.RandomNet(), n_agents=1000), 'mixing'),
    names      = (lambda: ss.Sim(unnamed(False)),                    lambda: ss.Sim(unnamed(True)),                        'name built from'),
    pop        = (lambda: ss.Sim(plain_sir(), stop=50),             lambda: ss.Sim(plain_sir(), total_pop=1000, stop=50), 'people'),
    stop       = (lambda: ss.Sim(plain_sir(), total_pop=1000),      lambda: ss.Sim(plain_sir(), total_pop=1000, stop=50), 'to 50'),
    acquisition = (lambda: corpus.malaria(),                         lambda: ss.Sim(plain_sir()),                         'acquisition'),
    midpoint   = (lambda: ss.Sim(fitted_sir(ss.par(bounds=[0.1, 0.5]))), lambda: ss.Sim(fitted_sir(ss.par(0.3, bounds=[0.1, 0.5]))), 'middle of its bounds'),
)
not_marked = dict() # Guesses that describe() does not mark yet, with the reason


@pytest.mark.parametrize('row', [pytest.param(row, marks=pytest.mark.xfail(reason=not_marked[row], strict=True)) if row in not_marked else row for row in guesses])
def test_guesses_marked(row):
    sc.heading(f'Testing that the guess of "{row}" is marked, and only when it is made')
    make_guess, make_given, key = guesses[row]
    marked = guessed(make_guess().describe(output=True))
    unmarked = guessed(make_given().describe(output=True))
    assert any(key in line for line in marked), f'Expected a line with "{key}" marked as guessed, but the marked lines are:\n' + '\n'.join(marked)
    assert not any(key in line for line in unmarked), f'Expected no line with "{key}" marked as guessed, but:\n' + '\n'.join(unmarked)
    return marked


# %% Streams are independent

class Counter(ss.Analyzer):
    """ An analyzer that changes nothing """
    compartmental = True

    def step(self):
        pass


@pytest.mark.parametrize('solver', stochastic)
def test_streams(solver):
    sc.heading(f'Testing that random streams are independent under {solver}')
    agents = dict(n_agents=2000) if solver == 'abm' else {}
    two_sirs = lambda *names: ss.Sim([corpus.make_sir(name=name) for name in names], solver=solver, unit='day', stop=100, total_pop=10_000, verbose=0, **agents) # Two SIRs that share no states
    alone = corpus.sir(solver=solver, unit='day', **agents).run().results.sir.I.values

    # An analyzer, or an unrelated disease, does not change the draws of a disease, whether the run is whole or stepped
    withan = corpus.sir(solver=solver, unit='day', analyzers=Counter(), **agents).run().results.sir.I.values
    assert np.array_equal(alone, withan), 'An analyzer changed the draws'
    if solver != 'abm': # Nor those of its outcomes (cases, hospitalizations, deaths), which a compartmental run draws after its states
        plain, counted = [corpus.outcomes(solver=solver, rand_seed=5, **kw).run().results.sir for kw in [{}, dict(analyzers=Counter())]]
        for key in ['cases', 'hosp', 'n_hosp', 'deaths']:
            assert np.array_equal(plain[key].values, counted[key].values), f'An analyzer changed the draws of the outcome "{key}" by up to {np.abs(plain[key].values - counted[key].values).max():n}'
    for profile in [False, True]: # Profiling makes the run step from Python
        both = two_sirs('sir', 'other').run(profile=profile).results.sir.I.values
        assert np.array_equal(alone, both), f'Another disease changed the draws (profile={profile}): max difference {np.abs(alone - both).max():n}'

    # Two identical diseases whose names are anagrams have their own streams
    ab = two_sirs('ab', 'ba').run().results
    assert not np.array_equal(ab.ab.I.values, ab.ba.I.values), 'Two modules got the same stream'

    if solver != 'abm':
        # The order the states are declared in does not change the draws
        def declared(order):
            states = dict(zip(order.split(), ss.states(order)))
            S, I, R, V = [states[name] for name in 'SIRV']
            sir = ss.Disease(infection=S >> I @ ss.perday(0.42) * I, recovery=I >> R @ ss.perday(0.09), vax=S >> V @ ss.perday(0.011), breakthrough=V >> I @ ss.perday(0.17) * I, init_n=dict(I=25), name='sir')
            return ss.Sim(sir, total_pop=10_000, stop=60, unit='day', solver=solver, rand_seed=587, verbose=0).run().results.sir.I.values
        siv, svi = declared('S I R V'), declared('S V I R')
        assert np.array_equal(siv, svi), f'Declaring the states in another order changed n_I by up to {np.abs(siv - svi).max():n}' # 155 under binomial before

        # A rate that varies in time takes the same values in a run stepped from Python as in a whole one, so the same draws
        def seasonal(solver):
            S, I, R, V = ss.states('S I R V')
            sir = ss.Disease(infection=S >> I @ ss.seasonal(mean=0.421, amplitude=0.46, peak=0.92) * I, recovery=I >> R @ ss.perday(0.0909), vax=S >> V @ ss.perday(0.011), init_n=dict(I=25), name='sir')
            return ss.Sim(sir, total_pop=100_000, stop=60, unit='day', solver=solver, rand_seed=587, verbose=0)
        for s in [solver, 'ode']:
            whole, stepped = [seasonal(s).run(profile=profile).results.sir.vax.values for profile in [False, True]]
            if s == 'ode': # With more than one substep, the kernel compiled into a whole run and the one called from Python round differently, by about 1e-16
                assert np.allclose(whole, stepped, rtol=1e-12, atol=0), f'{s}: stepping changed the vaccinated by up to {np.abs(whole - stepped).max():n}'
            else:
                assert np.array_equal(whole, stepped), f'{s}: stepping changed the vaccinated by up to {np.abs(whole - stepped).max():n}' # 61 under binomial before

        # Each run of several draws its outcomes from its own seed, as it would on its own
        sim = corpus.outcomes(solver=solver, rand_seed=5)
        runs = sim.run(n_runs=3).results.sir.cases.values
        for i in range(3):
            one = corpus.outcomes(solver=solver, rand_seed=5 + i).run().results.sir.cases.values
            assert np.array_equal(runs[i], one), f'Run {i} of three drew its outcomes differently from the same run on its own'
    return alone


def test_streams_across_sims():
    sc.heading('Testing that interleaving two sims does not change either')
    a1 = corpus.sir(solver='binomial', unit='day', rand_seed=1).run().results.sir.I.values
    a, b = corpus.sir(solver='binomial', unit='day', rand_seed=1), corpus.sir(solver='binomial', unit='day', rand_seed=2)
    a.run(until=50)
    b.run(until=50)
    a2 = a.run().results.sir.I.values
    assert np.array_equal(a1, a2), f'Running another sim in between changed the results by up to {np.abs(a1 - a2).max():n}'
    return a2


def test_crn_pairing():
    sc.heading('Testing that common random numbers pair the runs of a scenario and its baseline, for a declared disease as agents')
    def infections(seed, campaign):
        intv = ss.Delivery(move=dict(vaccinated=True), coverage=0.05, at=[10], name='campaign') if campaign else None # A small change
        return corpus.vaccination(solver='abm', interventions=intv, n_agents=2000, rand_seed=seed).run().results.sir.infection.values.sum()
    base, alt = [np.array([infections(seed, campaign) for seed in range(20)]) for campaign in [False, True]]
    ratio = np.var(alt - base)/(np.var(alt) + np.var(base)) # 1 for independent runs
    assert ratio < 0.5, f'The difference between the scenarios varies {ratio:.2f} times as much as for independent runs' # 0.14 to 0.32 over four sets of 20 seeds
    return ratio


# %% The code of a model builds it again

@pytest.mark.parametrize('name', corpus.models.keys())
def test_code_round_trip(name, tmp_path):
    sc.heading(f'Testing that the code written for the corpus model "{name}" builds the same model, or says what it leaves out')
    sim = corpus.models[name]()
    path = tmp_path/'model.py'
    sim.to_code(path)
    if 'Not written as code' in path.read_text(): # A network, a function, data, or a module written as code, listed to add by hand
        return
    rebuilt = runpy.run_path(str(path))['sim'] # From a file, as a user runs it, so the source of a function in it can be read
    assert rebuilt.hash() == sim.hash(), f'{name}: the code builds a different model'
    return


# %% The solvers agree

fine = dict(dt=ss.days(0.25), total_pop=10_000) # Agents and stepped solvers differ from the ODE by an amount of order dt, so compare at a fine step; and at a size that runs quickly event by event
skip_solver = dict( # Corpus models that a solver cannot run, with the reason
    screening = dict(gillespie='rounds of a delivery need a step'),
)
loose = dict( # Wider tolerances, with the reason
    sir         = 0.07, # Agents are about 3% above the ODE in S and below it in R, and the mean of 5 runs varies by about 2%
    custom      = 0.10, # Few people are hospitalized, so the counts are noisy
    threshold   = 0.15, # The switch on and off at the threshold amplifies differences
    ages        = 0.10, # Each of 5,000 agents is 20 people, so the few who start infected make the timing of the epidemic noisy (3% with 50,000)
    age_vaccine = 0.12, # And a force of infection near 0.7 per day, so the difference of order dt is about 6% even at a quarter-day step (1% at a tenth)
    patches     = 0.10,
)


def mean_states(sim):
    """ The mean over time of each state, for each declared module """
    return {f'{mod.name}.{key}':res.values.mean() for mod in sim.declared for key,res in mod.results.items() if key in mod.graph.states and isinstance(res, ss.Result) and np.ndim(res.values) == 1}


simple = ['sir', 'seir', 'erlang', 'branching', 'ages'] # The models marked simple in 18-conformance.md
slow = bool(os.getenv('STARSIM_SLOW')) # Set to check every model of the corpus


@pytest.mark.parametrize('name', corpus.compartmental if slow else simple)
def test_solvers_agree(name):
    sc.heading(f'Testing that every solver agrees with the ODE for the corpus model "{name}"')
    make = corpus.models[name]
    ode = mean_states(make(solver='ode', **fine).run())
    tol = loose.get(name, 0.05) # Within 5%, which allows for the noise of a few runs
    floor = 0.1*max(ode.values()) # A small state is compared with a tenth of the largest, so its noise does not fail it
    for solver in stochastic:
        if solver in skip_solver.get(name, {}):
            continue
        runs = [mean_states(make(solver=solver, rand_seed=seed, **fine).run()) for seed in range(5 if solver == 'abm' else 3)] # Agents vary more between runs, e.g. in when the epidemic takes off
        for key,val in ode.items():
            mean = np.mean([run[key] for run in runs])
            assert abs(mean - val) <= tol*max(val, floor), f'{name} under {solver}: {key} averages {mean:n}, against {val:n} for the ODE'
    return ode


def test_binomial_short_stay():
    sc.heading('Testing that the chain binomial follows the ODE when a stay is one step')
    def make(solver, seed=0):
        S, I, R = ss.states('S I R')
        dis = ss.Disease(S, I, R, infection=S >> I @ ss.permonth(2)*I.frac, recovery=I >> R @ ss.months(1), init=dict(I=0.001), name='sir')
        return ss.Sim(dis, solver=solver, unit='month', stop=24, total_pop=1e6, rand_seed=seed, verbose=0).run().results.sir
    ode = make('ode')
    runs = [make('binomial', seed) for seed in range(3)]
    final = np.mean([res.R[-1] for res in runs])
    curve = np.mean([res.I.values for res in runs], axis=0)
    assert np.isclose(final, ode.R[-1], rtol=0.05), f'The chain binomial ends with {final:n} recovered, against {ode.R[-1]:n} for the ODE' # 950,577 against 797,129 with whole steps
    assert np.abs(curve - ode.I.values).max() < 0.2*ode.I.max(), 'The chain binomial\'s infected are far from the ODE\'s' # 1.33 of the peak with whole steps
    return


def test_agents_match_ode():
    sc.heading('Testing that agents match the ODE on the spec\'s models, and that R0() is the analytic value')
    analytic = dict( # From the corpus's parameters
        sir     = 0.3*10, # β·D
        seir    = 0.3*10, # β·D, since everyone exposed becomes infectious
        malaria = np.sqrt(0.3*0.5*50_000/10_000*100 * 0.3*0.5*0.1/(0.1 + 0.1)*10), # √(abM/(Nr) · acσ/((σ + μ)μ)) = 7.5, with M = 50,000 mosquitoes at equilibrium and N = 10,000 people (was 0)
    )
    for name,R0 in analytic.items():
        sim = corpus.models[name](solver='ode', **fine)
        ode = mean_states(sim.run())
        assert np.isclose(sim.diseases[name].R0(), R0), f'{name}: R0() is {sim.diseases[name].R0():n}, not {R0:n}'
        if name in simple: # Agents are compared for these in test_solvers_agree()
            continue
        runs = [mean_states(corpus.models[name](solver='abm', rand_seed=seed, **fine).run()) for seed in range(3)]
        floor = 0.1*max(ode.values())
        for key,val in ode.items():
            mean = np.mean([run[key] for run in runs])
            assert abs(mean - val) <= 0.03*max(val, floor), f'{name} as agents: {key} averages {mean:n}, against {val:n} for the ODE' # About 1% apart at a quarter-day step
    return


@pytest.mark.parametrize('solver', ['ode', 'binomial', 'gillespie', 'tau', 'abm'])
def test_exit_order(solver):
    sc.heading(f'Testing that the order of competing exits does not matter under {solver}')
    shares = []
    for order in [('a', 'b', 'c'), ('c', 'a', 'b')]:
        I = ss.State(name='I')
        dests = {key:ss.State(name=key.upper()) for key in 'abc'}
        exits = {key:I >> dests[key] @ ss.perday(1.0) for key in order} # Three equal exits, fast compared with the step
        dis = ss.Disease(I, *dests.values(), **exits, first='I', name='exits')
        agents = dict(n_agents=20_000) if solver == 'abm' else {}
        sim = ss.Sim(dis, unit='day', stop=5, solver=solver, total_pop=100_000, verbose=0, **agents).run()
        assert sim.solver == solver
        res = sim.results.exits
        left = np.array([res[f'{k.upper()}'][-1] for k in 'abc'])
        shares.append(left/left.sum())
    for share in shares:
        assert np.allclose(share, 1/3, atol=0.02), f'Expected a third by each exit, not {share}'
    return shares


def test_dwell_forms():
    sc.heading('Testing that agents agree with the ODE at a fine step for every form of dwell time')
    forms = dict(
        exponential  = lambda I, R, D: dict(out=I >> R @ ss.days(4)),
        rate         = lambda I, R, D: dict(out=ss.Transition(I, R, rate=ss.perday(0.25))),
        fixed        = lambda I, R, D: dict(out=ss.Transition(I, R, dur=ss.constant(ss.days(4)))),
        erlang       = lambda I, R, D: dict(out=ss.Transition(I, R, dur=ss.days(4), stages=3)),
        lognormal    = lambda I, R, D: dict(out=ss.Transition(I, R, dur=ss.lognorm_ex(ss.days(4), ss.days(2)))),
        branches     = lambda I, R, D: dict(a=ss.Transition(I, R, p=0.7, dur=ss.days(4)), b=ss.Transition(I, D, p=0.3, dur=ss.days(2))), # Decided on entry
        destinations = lambda I, R, D: dict(out=ss.Transition(I, [R, D], p=[0.7, 0.3], dur=ss.days(4))), # Decided at exit
        competing    = lambda I, R, D: dict(out=ss.Transition(I, R, dur=ss.constant(ss.days(4))), die=ss.Transition(I, D, rate=ss.perday(0.1))),
        after        = lambda I, R, D: dict(out=ss.Transition(I, R, rate=ss.perday(0.5), after=ss.days(2))), # A minimum stay, then a rate
    )
    def run(form, solver):
        I, R, D = ss.states('I R D')
        dis = ss.Disease(I, R, D, **forms[form](I, R, D), first='I', name='cohort')
        size = dict(n_agents=20_000) if solver == 'abm' else dict(total_pop=20_000)
        res = ss.Sim(dis, unit='day', dt=0.05, stop=15, solver=solver, rand_seed=1, verbose=0, **size).run().results.cohort
        return np.array([res.I.values.mean(), res.R[-1], res.D[-1]])/20_000 # Time infected, and where people end up
    for form in forms:
        ode, abm = run(form, 'ode'), run(form, 'abm')
        assert np.allclose(abm, ode, rtol=0.03, atol=0.005), f'{form}: agents {abm}, ODE {ode}'
    return


def test_wide_dwell():
    sc.heading('Testing that a dwell time too variable for Erlang stages runs as agents under auto, and warns under the ODE')
    def make(c, solver='auto'):
        A, B = ss.states('A B')
        dis = ss.Disease(A, B, go=A >> B @ ss.peryear(1), back=ss.Transition(B, A, dur=ss.weibull(c=c, scale=ss.years(1))), first='A', name='m')
        return ss.Sim(dis, total_pop=1000, stop=2010, solver=solver, verbose=0).init()
    assert make(0.5).solver == 'ode' # A coefficient of variation of 2.2
    assert make(0.3).solver == 'abm' # Of 5.4: one exponential stage at its mean of 9.3 years kept nearly everyone in B
    with pytest.warns(RuntimeWarning, match='coefficient of variation'):
        make(0.3, 'ode')
    return


def test_destinations():
    sc.heading('Testing that every solver agrees on the flow to each destination, by age')
    def run(solver, seed=1):
        I, R, D = ss.states('I R D')
        dis = ss.Disease(I, R, D, out=ss.Transition(I, [R, D], p=[0.7, 0.3], dur=ss.days(4)), init_n=dict(I=1000), first='I', name='cohort')
        dis.stratify(age=ss.years([0, 40]))
        res = ss.Sim(dis, n_agents=1000, unit='day', stop=40, solver=solver, rand_seed=seed, verbose=0).run().results.cohort
        return np.array([res.out.D.select(age=age).sum() for age in [0, 1]])
    ode = run('ode')
    for solver in stochastic:
        runs = np.array([run(solver, seed) for seed in range(20)])
        se = runs.std(axis=0)/np.sqrt(len(runs))
        assert np.all(abs(runs.mean(axis=0) - ode) <= 3*se + 1e-9), f'{solver}: deaths by age {runs.mean(axis=0)}, against {ode} for the ODE'
    return ode


# %% Conservation and exact identities

closed = ['sir', 'seir', 'erlang', 'lognormal', 'ages', 'patches', 'vaccination', 'screening', 'flu_rsv', 'age_vaccine', 'patch_matrix'] # No births, deaths, or flows to ss.null


@pytest.mark.parametrize('name', closed)
def test_conservation(name):
    sc.heading(f'Testing that the corpus model "{name}" keeps its population under every solver')
    for solver in ['ode', 'binomial', 'gillespie', 'abm']:
        if solver in skip_solver.get(name, {}):
            continue
        sim = corpus.models[name](solver=solver, total_pop=10_000).run()
        for mod in sim.declared:
            total = sum(mod.results[f'{state.name}'].values for state in mod.states.values() if state.people)
            assert np.allclose(total, total[0], rtol=1e-9), f'{name} under {solver}: the population of {mod.name} goes from {total[0]:n} to {total[-1]:n}'
    return


@pytest.mark.parametrize('dwell', ['exponential', 'fixed', 'erlang', 'lognormal', 'after'])
def test_crossing_conservation(dwell):
    sc.heading(f'Testing that diseases tracked jointly keep their people under every solver, with a dwell that is {dwell}')
    dwells = dict(
        exponential = lambda P, NP: ss.Transition(P, NP, dur=ss.months(9)),
        fixed       = lambda P, NP: ss.Transition(P, NP, dur=ss.constant(ss.months(9))), # A conveyor: the ODE lost 9% of people
        erlang      = lambda P, NP: ss.Transition(P, NP, dur=ss.months(9), stages=3),
        lognormal   = lambda P, NP: ss.Transition(P, NP, dur=ss.lognorm_ex(ss.months(9), ss.months(3))),
        after       = lambda P, NP: ss.Transition(P, NP, rate=ss.peryear(2), after=ss.months(3)),
    )
    for solver in ['ode', 'binomial', 'abm']:
        M = ss.states('M0 M1 M2')
        m = ss.Disease(*M, **{f'go{i}':M[i] >> M[(i+1) % 3] @ ss.peryear(1.0) for i in range(3)}, first='M0', name='m')
        NP, P = ss.states('NP P')
        p = ss.Disease(NP, P, conception=NP >> P @ ss.peryear(0.25), birth=dwells[dwell](P, NP), first='NP', name='p')
        mod = ss.modify(p.conception, multiply=0.1, where=M[1]) # Crosses the two
        size = dict(n_agents=2000) if solver == 'abm' else dict(total_pop=2000)
        sim = ss.Sim([m, p], connectors=mod, start=2000, stop=2010, dt=ss.months(1), solver=solver, verbose=0, **size).run()
        assert solver == 'abm' or sim.crossings
        for mod in sim.declared:
            total = sum(mod.results[f'{state.name}'].values for state in mod.states.values() if state.people)
            assert np.allclose(total, 2000, rtol=1e-9), f'{dwell} under {solver}: the population of {mod.name} goes from {total[0]:n} to {total[-1]:n}'
    return


def test_empty_place():
    sc.heading('Testing that a place with no one in it changes nothing, under every solver that runs by group')
    def make(solver, places=True):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.2)*I.frac + ss.perday(1e-5)*I.n + ss.perday(1e-6)*I.total), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
        if places:
            sir.stratify(place=['a', 'b'], kind='space', split=[1.0, 0.0])
        return ss.Sim(sir, solver=solver, total_pop=10_000, unit='day', stop=60, verbose=0)
    one = make('ode', places=False).run().results.sir.R.values
    for solver in ['ode', 'jax', 'binomial', 'tau', 'gillespie']:
        sim = make('ode' if solver == 'jax' else solver).init()
        sim.jax = solver == 'jax'
        res = sim.run().results.sir
        assert np.all(np.isfinite(res.I.values)) and res.I.select(place='b').values.max() == 0, solver
        if solver in ['ode', 'jax']:
            assert np.allclose(res.R.values, one, rtol=1e-9), f'{solver}: an empty place changes the epidemic'
    return


def test_identities():
    sc.heading('Testing constant-rate decay and a fixed branch share against their exact values')

    # Constant-rate decay is exponential
    I, R = ss.states('I R')
    decay = ss.Disease(recovery=ss.Transition(I, R, rate=ss.perday(0.1)), first='I', name='decay')
    res = ss.Sim(decay, total_pop=1e6, unit='day', stop=50, solver='ode', verbose=0).run().results.decay
    t = np.arange(51)
    assert np.allclose(res.I.values, 1e6*np.exp(-0.1*t), rtol=1e-5) # RK4's error is about 1e-7 a step at this rate

    # A fixed branch share is the share that leaves by each branch
    for solver in ['ode', 'binomial', 'abm']:
        I, R, D = ss.states('I R D')
        branch = ss.Disease(course=ss.Transition(I, [R, D], dur=5, p=[None, 0.2]), first='I', name='branch')
        agents = dict(n_agents=50_000) if solver == 'abm' else {}
        res = ss.Sim(branch, total_pop=1e6, unit='day', stop=100, solver=solver, verbose=0, **agents).run().results.branch
        share = res.D[-1]/(res.R[-1] + res.D[-1])
        atol = 1e-9 if solver == 'ode' else 0.01 # Exact for the ODE; for a stochastic run, many standard errors of the share
        assert np.isclose(share, 0.2, atol=atol), f'{solver}: a share of {share:n} left by D, not 0.2'

    # A fixed delay is a conveyor: everyone who starts in E leaves exactly 10 days later, under every solver and as agents
    E, I = ss.states(2)
    for solver in ['ode', 'binomial', 'gillespie', 'abm']:
        size = dict(n_agents=1000) if solver == 'abm' else dict(total_pop=1000)
        cohort = ss.Disease(onset=E >> I @ ss.constant(ss.days(10)), first='E', name='cohort')
        n_E = ss.Sim(cohort, unit='day', stop=20, solver=solver, verbose=0, **size).run().results.cohort.E.values
        assert n_E[10] == 1000 and n_E[11] == 0, f'{solver}: {n_E[10]:n} in E on day 10 and {n_E[11]:n} on day 11, not 1000 and 0'

    # With a constant inflow as well, E holds the inflow over the delay: exactly in the ODE at every step, and on average in the stochastic solvers
    flow = ss.Disease(arrival=ss.null >> E @ ss.perday(100), onset=E >> I @ ss.constant(ss.days(10)), first='E', name='flow')
    exact = np.r_[1000 + 100*np.arange(11), np.full(20, 1000)] # Those who started in E leave on day 10, and the arrivals of each day 10 days later
    n_E = ss.Sim(flow, total_pop=1000, unit='day', stop=30, verbose=0).run().results.flow.E.values
    assert np.allclose(n_E, exact, rtol=1e-9)
    for solver in ['binomial', 'gillespie']:
        runs = ss.Sim(flow, total_pop=1000, unit='day', stop=30, solver=solver, verbose=0).run(n_runs=20).results.flow.E.values
        assert np.isclose(np.mean(runs, axis=0)[11:].mean(), 1000, rtol=0.03), f'{solver}: E holds {np.mean(runs, axis=0)[11:].mean():n} on average, not 1000'
    return


def test_acquisition():
    sc.heading('Testing that an infection by a bite rate is an acquisition: the bednet idiom, a vaccine, and a cure act on the Ross-Macdonald model')
    for solver,size in [('ode', dict(total_pop=5000)), ('abm', dict(n_agents=5000))]:
        infections = {}
        for key,intv in dict(none=None, bednet=ss.modify('malaria.rel_sus', multiply=0.0), vaccine=ss.vaccinate(ss.Vaccine(efficacy=1.0), coverage=1.0, at=[0]),
                             cure=ss.treat(ss.Treatment(cure=1.0), coverage=0.9, at=[100], name='mda')).items():
            sim = corpus.malaria(solver=solver, interventions=intv, **size).run()
            infections[key] = sim.results.malaria.infection.values.sum()
        assert infections['none'] > 5000 and infections['bednet'] == 0 # Was the same: rel_sus did not reach an infection by Iv.n
        assert infections['vaccine'] == 0
        assert sim.results.mda.delivered.values.sum() > 1000 # Was 0: nobody counted as infected
    def imported(acq): # An importation alongside transmission is not an acquisition unless it says so
        S, I, R = ss.states(3)
        return ss.Disease(infection=S >> I @ ss.perday(0.2)*I, imported=ss.Transition(S, I, rate=ss.perday(0.01), acquisition=acq), recovery=I >> R @ ss.days(10), name='sir')
    protected = [ss.Sim(imported(acq), interventions=ss.modify('sir.rel_sus', multiply=0), unit='day', stop=50, total_pop=1000, verbose=0).run().results.sir.imported.values.sum() for acq in [None, True]]
    assert protected[0] > 300 and protected[1] == 0
    return


# %% Behavior: a regression, a counter, and a table over the destinations

def behavior(solver, seed=1, n=2000):
    """ A choice of method: a regression over two boolean dimensions and a counter, and a switching row by age and destination (a clock since the last birth is agents only: see test_dimensions.py::test_clocks) """
    from scipy.special import expit
    none, pill, iud, P = ss.states('none pill iud P')
    p_use = expit(-1.0 + 0.5*ss.ppl.urban + 0.3*ss.ppl.edu + 0.4*ss.ppl.parity) # Parity is a counter, so is exact as compartments
    mix = ss.par(ss.by(dict(age=[0, 0, 30, 30], to=['pill', 'iud', 'pill', 'iud'], value=[0.7, 0.3, 0.4, 0.6]))) # A row of the switching matrix by age, over the destinations
    contra = ss.Disease(none, pill, iud, P,
        start    = ss.Transition(none, [pill, iud], rate=ss.peryear(0.3)*p_use, p=[mix]),
        stop     = ss.Transition(pill, none, rate=ss.peryear(0.2)),
        remove   = ss.Transition(iud, none, rate=ss.peryear(0.1)),
        conceive = ss.Transition(none, P, rate=ss.peryear(0.2)),
        deliver  = ss.Transition(P, none, dur=ss.months(9), increment='parity'),
        mix=mix, first='none', name='contra')
    size = dict(n_agents=n) if solver == 'abm' else dict(total_pop=n)
    sim = ss.Sim(contra, solver=solver, rand_seed=seed, start=2000, stop=2006, dt=ss.months(1), use_aging=False, verbose=0, **size)
    sim.stratify(age=[0, 30])
    sim.stratify(urban=[False, True], split=[0.4, 0.6])
    sim.stratify(edu=[False, True], split=[0.5, 0.5])
    sim.stratify(parity=[0, 1, 2, 3])
    return sim


def behavior_shares(sim):
    """ The share of people in each state (none, pill, iud, P) by age group, over the last year """
    res = sim.results.contra
    n = np.array([res[s].by('age').values[-12:].mean(axis=0) for s in ['none', 'pill', 'iud', 'P']])
    return n/n.sum(axis=0)


@sc.timer()
def test_behavior_solvers():
    sc.heading('Testing that a module of behavior (a regression, a counter, and a table over the destinations) has the same shares by method and age under every solver')
    ode = behavior_shares(behavior('ode').run())
    sim = behavior('binomial', n=1_000_000)
    sim.pars.substeps = 10 # Whole steps of the chain binomial would lengthen the 9 months pregnant by half a step (6% more pregnant)
    binomial = behavior_shares(sim.run())
    assert np.allclose(binomial, ode, rtol=0.01), f'Expected the binomial solver to match the ODE: {binomial} against {ode}'
    for solver in ['gillespie', 'abm']: # 20 runs each, within 3 standard errors
        runs = np.array([behavior_shares(behavior(solver, seed=seed).run()) for seed in range(20)])
        mean, se = runs.mean(axis=0), runs.std(axis=0, ddof=1)/np.sqrt(len(runs))
        assert np.all(np.abs(mean - ode) < 3*se), f'Expected {solver} to match the ODE within 3 standard errors: {mean} against {ode}, z = {(mean - ode)/se}'
    return ode


# %% Code beside a declared model

class Watch(ss.Analyzer):
    """ An analyzer that only reads a count, saying nothing about solvers """
    def init_post(self):
        super().init_post()
        self.seen = []
        return

    def step(self):
        self.seen.append(self.sim.diseases.sir.I.n)
        return


class NumpySIS(ss.Module):
    """ An ODE of its own, in plain NumPy, with Euler steps of a day """
    def init_post(self):
        super().init_post()
        self.y = np.array([0.99, 0.01])
        return

    def step(self):
        S, I = self.y
        flow = (0.3*S*I - 0.1*I)*self.t.dt.days
        self.y = np.array([S - flow, I + flow])
        return


class Tagger(ss.Intervention):
    """ An intervention written as code, which works on agents """
    def step(self):
        return self.sim.diseases.sir.S.uids


def test_coded_beside():
    sc.heading('Testing that code beside a declared model does not change its solver unless it needs agents')
    alone = corpus.sir(unit='day').run()
    assert alone.solver == 'ode'

    # An analyzer that reads counts, or a NumPy ODE of its own, runs beside the compartments, and the declared model's results are unchanged
    watched = corpus.sir(unit='day', analyzers=Watch()).run()
    assert watched.solver == 'ode', f'An analyzer changed the solver to {watched.solver}'
    assert np.array_equal(watched.results.sir.I.values, alone.results.sir.I.values)
    assert np.allclose(watched.analyzers.watch.seen, alone.results.sir.I.values[:len(watched.analyzers.watch.seen)])
    numpy = corpus.sir(unit='day', connectors=NumpySIS()).run()
    assert numpy.solver == 'ode' and np.array_equal(numpy.results.sir.I.values, alone.results.sir.I.values)
    assert numpy.connectors.numpysis.y[1] > 0.5 # Its own model ran (to near 1 - 0.1/0.3)

    # Code that says it needs agents makes the model run as agents; an intervention written as code does not, and if it reads agents, the run stops and says so
    class Agentish(Watch):
        compartmental = False
    class AgentTagger(Tagger):
        compartmental = False
    for mod in [Agentish(), AgentTagger()]:
        sim = corpus.sir(unit='day', **{'analyzers' if isinstance(mod, ss.Analyzer) else 'interventions':mod}).init()
        assert sim.solver == 'abm', f'{type(mod).__name__} should need agents'
    assert corpus.sir(unit='day', interventions=Tagger()).init().solver == 'ode' # Written as code is not a reason for agents
    with pytest.raises(ss.CapabilityError, match='compartmental = False'):
        corpus.sir(unit='day', interventions=Tagger()).run()
    with pytest.raises(ss.CapabilityError):
        corpus.sir(unit='day', interventions=AgentTagger(), solver='ode').init()
    class Level(ss.Module): # A plain module that keeps a value for each person needs agents (it ran as an ODE and failed partway)
        def __init__(self):
            super().__init__()
            self.define_states(ss.FloatArr('level', default=0.0))
    assert corpus.sir(unit='day', connectors=Level()).init().solver == 'abm'
    return numpy


def test_joint_population():
    sc.heading('Testing diseases tracked jointly: the same people in each, as a joint model written by hand, within a budget')
    def pair(solver='auto', **kwargs):
        S, I = ss.states(2)
        a = ss.Disease(S, I, infection=S >> I @ ss.peryear(0.5)*I, death=ss.Transition(I, ss.null, dur=2), init_prev=0.1, name='a')
        S, I = ss.states(2)
        b = ss.Disease(S, I, infection=S >> I @ ss.peryear(0.3)*I, recovery=ss.Transition(I, S, dur=3), init_prev=0.1, name='b')
        return ss.Sim([a, b], total_pop=20_000, stop=2030, solver=solver, verbose=0, **kwargs)

    # Dying of one disease leaves the other too, though neither refers to the other (b kept all 20,000 people, with 1,642 left in a)
    for solver in ['ode', 'binomial']:
        sim = pair(solver).run()
        res = sim.results
        assert sim.crossings and np.allclose(res.a.S + res.a.I, res.b.S + res.b.I)

    # The same as the joint model written as one disease, whose states are the combinations
    SS, SI, IS, II = ss.states('SS SI IS II')
    ba, bb = ss.peryear(0.5), ss.peryear(0.3)
    joint = ss.Disease(SS, SI, IS, II, a1=SS >> IS @ ba*(IS + II), a2=SI >> II @ ba*(IS + II), b1=SS >> SI @ bb*(SI + II), b2=IS >> II @ bb*(SI + II),
                       r1=SI >> SS @ ss.years(3), r2=II >> IS @ ss.years(3), d1=ss.Transition(IS, ss.null, dur=2), d2=ss.Transition(II, ss.null, dur=2),
                       init=dict(SS=0.81, SI=0.09, IS=0.09, II=0.01), first='SS', name='ab')
    j = ss.Sim(joint, total_pop=20_000, stop=2030, verbose=0).run().results.ab
    res = pair().run().results
    assert np.allclose(res.a.I.values, j.IS.values + j.II.values, rtol=1e-9) and np.allclose(res.b.I.values, j.SI.values + j.II.values, rtol=1e-9)

    # Above the budget, auto runs agents, and a compartmental solver that was asked for stops before anything is made
    with ss.options.context(max_joint=3):
        sim = pair().init()
        assert sim.solver == 'abm' and 'max_joint' in sim.solver_reason
        with pytest.raises(ss.CapabilityError, match='max_joint=4'):
            pair('ode').init()
    S, A, B, I = ss.states('S A B I') # A first state that splits on entry has no joint form yet: auto runs agents rather than stopping
    split = ss.Disease(S, A, B, I, split=ss.Transition(S, [A, B], p=[0.5, None], dur=0), inf=A >> I @ ss.peryear(0.5)*I, init=dict(I=0.1), name='a')
    assert ss.Sim([split, ss.SIS(name='b')], demographics=ss.death(rate=ss.peryear(0.01)), total_pop=1000, stop=2010, verbose=0).init().solver == 'abm'
    text = pair().init().describe(output=True)
    assert 'a × b: 2 × 2 = 4 joint states of people' in text and 'people die of a' in text and 'start independent' in text
    assert 'plan         a, b: as compiled code, the whole run in one call; a × b as one program' in text
    return j


if __name__ == '__main__':
    T = sc.timer()
    for row in guesses:
        try:
            test_guesses_marked(row)
        except AssertionError as e:
            print(f'{row}: {e}')
    for name in corpus.compartmental:
        test_solvers_agree(name)
    behavior = test_behavior_solvers()
    joint = test_joint_population()
    T.toc()
