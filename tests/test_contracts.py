"""
Contract tests: properties every model and solver must have, checked on the model corpus
(see corpus.py). A failure here is usually a silent wrong answer somewhere else.
"""

# %% Imports and settings
import os
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import corpus

stochastic = ['binomial', 'gillespie', 'abm']


def people_states(sim, mod):
    """ The names of the states of a declared module that are people """
    return [state.name for state in mod.graph.people_states]


# %% The guesses are marked

def guessed(text):
    """ The lines of describe() marked as guesses """
    return [line for line in text.splitlines() if '[guessed' in line or 'inferred from' in line or 'nothing in the model has a unit' in line]


def string_sir(**kwargs):
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), name='sir', **kwargs)


def declared_sir(**kwargs):
    S, I, R = ss.State(name='S'), ss.State(name='I', infectious=True), ss.State(name='R')
    return ss.Disease(S, I, R, infection=S >> I @ 0.3*I, recovery=I >> R @ ss.erlang(mean=10, k=3, unit='days'), init_prev=0.01, name='sir', **kwargs)


def water(declared):
    S, I, R = ss.states('S I R')
    W = ss.State(name='W', people=False) if declared else ss.State(name='W')
    return ss.Disease(S >> I @ ss.perday(0.5)*W.n/(W.n + 1e5), I >> R @ ss.days(5), ss.null >> W @ 10*I.n, W >> ss.null @ ss.days(30), init=dict(I=0.001), name='cholera')


def lognormal(stages):
    dur = ss.lognorm_ex(mean=10, std=4, unit='days')
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=dur, stages=stages), init_prev=0.01, name='sir')


def unnamed(named):
    S, I, R = ss.states('S I R')
    infection, recovery = S >> I @ 0.3*I, I >> R @ ss.days(10)
    if named:
        return ss.Disease(infection, recovery, init_prev=0.01, name='sir')
    return ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init_prev=0.01, name='sir')


# Each row of the "what is guessed" table in 03-states-and-transitions.md: a sim that makes the guess, one that does not, and what describe() says
guesses = dict(
    states     = (lambda: ss.Sim(string_sir()),                      lambda: ss.Sim(declared_sir()),                       'inferred from the arrows'),
    source     = (lambda: ss.Sim(string_sir()),                      lambda: ss.Sim(declared_sir()),                       'source is the destination'),
    infectious = (lambda: ss.Sim(string_sir()),                      lambda: ss.Sim(declared_sir()),                       'I infectious'),
    init       = (lambda: ss.Sim(string_sir()),                      lambda: ss.Sim(string_sir(init_prev=0.01)),           'init:'),
    people     = (lambda: ss.Sim(water(False)),                      lambda: ss.Sim(water(True)),                          'a count, not people'),
    dwell      = (lambda: ss.Sim(string_sir()),                      lambda: ss.Sim(declared_sir()),                       'dwell ~'),
    stages     = (lambda: ss.Sim(lognormal(None), solver='ode'),     lambda: ss.Sim(lognormal(4), solver='ode'),           'stages'),
    unit       = (lambda: ss.Sim(string_sir()),                      lambda: ss.Sim(string_sir(), unit='day'),             'unit'),
    mixing     = (lambda: ss.Sim(string_sir()),                      lambda: ss.Sim(string_sir(), networks=ss.RandomNet(), n_agents=1000), 'mixing'),
    names      = (lambda: ss.Sim(unnamed(False)),                    lambda: ss.Sim(unnamed(True)),                        'name built from'),
    pop        = (lambda: ss.Sim(string_sir(), stop=50),             lambda: ss.Sim(string_sir(), total_pop=1000, stop=50), 'people'),
    stop       = (lambda: ss.Sim(string_sir(), total_pop=1000),      lambda: ss.Sim(string_sir(), total_pop=1000, stop=50), 'to 50'),
    acquisition = (lambda: corpus.malaria(),                         lambda: ss.Sim(string_sir()),                         'acquisition'),
    midpoint   = (lambda: ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=ss.par(bounds=[0.1, 0.5])), name='sir')),
                  lambda: ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=ss.par(0.3, bounds=[0.1, 0.5])), name='sir')), 'middle of its bounds'),
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

def two_sirs(names=('a', 'b'), solver='binomial', **kwargs):
    """ Two SIRs that share no states, in one sim """
    mods = [corpus.make_sir(name=name) for name in names]
    return ss.Sim(mods, solver=solver, total_pop=10_000, unit='day', stop=100, verbose=0, **kwargs)


class Counter(ss.Analyzer):
    """ An analyzer that changes nothing """
    compartmental = True

    def step(self):
        pass


@pytest.mark.parametrize('solver', stochastic)
def test_streams(solver):
    sc.heading(f'Testing that random streams are independent under {solver}')
    agents = dict(n_agents=2000) if solver == 'abm' else {}
    alone = corpus.sir(solver=solver, unit='day', **agents).run().results.sir.n_I.values

    # An analyzer, or an unrelated disease, does not change the draws of a disease, whether the run is whole or stepped
    withan = corpus.sir(solver=solver, unit='day', analyzers=Counter(), **agents).run().results.sir.n_I.values
    assert np.array_equal(alone, withan), 'An analyzer changed the draws'
    for profile in [False, True]: # Profiling makes the run step from Python
        both = ss.Sim([corpus.make_sir(), corpus.make_sir(name='other')], solver=solver, unit='day', stop=100, total_pop=10_000, verbose=0, **agents)
        both = both.run(profile=profile).results.sir.n_I.values
        assert np.array_equal(alone, both), f'Another disease changed the draws (profile={profile}): max difference {np.abs(alone - both).max():n}'

    # Two identical diseases whose names are anagrams have their own streams
    ab = two_sirs(('ab', 'ba'), solver=solver, **agents).run().results
    assert not np.array_equal(ab.ab.n_I.values, ab.ba.n_I.values), 'Two modules got the same stream'

    if solver != 'abm':
        # The order the states are declared in does not change the draws (C12a)
        def declared(order):
            states = dict(zip(order.split(), ss.states(order)))
            S, I, R, V = [states[name] for name in 'SIRV']
            sir = ss.Disease(infection=S >> I @ ss.perday(0.42) * I, recovery=I >> R @ ss.perday(0.09), vax=S >> V @ ss.perday(0.011), breakthrough=V >> I @ ss.perday(0.17) * I, init=dict(I=25), name='sir')
            return ss.Sim(sir, total_pop=10_000, stop=60, unit='day', solver=solver, rand_seed=587, verbose=0).run().results.sir.n_I.values
        siv, svi = declared('S I R V'), declared('S V I R')
        assert np.array_equal(siv, svi), f'Declaring the states in another order changed n_I by up to {np.abs(siv - svi).max():n}' # 155 under binomial before

        # A rate that varies in time takes the same values in a run stepped from Python as in a whole one, so the same draws (C12c)
        def seasonal(solver):
            S, I, R, V = ss.states('S I R V')
            sir = ss.Disease(infection=S >> I @ ss.seasonal(mean=0.421, amplitude=0.46, peak=0.92) * I, recovery=I >> R @ ss.perday(0.0909), vax=S >> V @ ss.perday(0.011), init=dict(I=25), name='sir')
            return ss.Sim(sir, total_pop=100_000, stop=60, unit='day', solver=solver, rand_seed=587, verbose=0)
        for s in [solver, 'ode']:
            whole, stepped = [seasonal(s).run(profile=profile).results.sir.vax.values for profile in [False, True]]
            if s == 'ode': # With more than one substep, the kernel compiled into a whole run and the one called from Python round differently, by about 1e-16
                assert np.allclose(whole, stepped, rtol=1e-12, atol=0), f'{s}: stepping changed the vaccinated by up to {np.abs(whole - stepped).max():n}'
            else:
                assert np.array_equal(whole, stepped), f'{s}: stepping changed the vaccinated by up to {np.abs(whole - stepped).max():n}' # 61 under binomial before

        # Each run of several draws its outcomes from its own seed, as it would on its own (C19)
        sim = corpus.outcomes(solver=solver, rand_seed=5)
        runs = sim.run(n_runs=3).results.sir.cases.values
        for i in range(3):
            one = corpus.outcomes(solver=solver, rand_seed=5 + i).run().results.sir.cases.values
            assert np.array_equal(runs[i], one), f'Run {i} of three drew its outcomes differently from the same run on its own'
    return alone


def test_streams_across_sims():
    sc.heading('Testing that interleaving two sims does not change either')
    a1 = corpus.sir(solver='binomial', unit='day', rand_seed=1).run().results.sir.n_I.values
    a, b = corpus.sir(solver='binomial', unit='day', rand_seed=1), corpus.sir(solver='binomial', unit='day', rand_seed=2)
    a.run(until=50)
    b.run(until=50)
    a2 = a.run().results.sir.n_I.values
    assert np.array_equal(a1, a2), f'Running another sim in between changed the results by up to {np.abs(a1 - a2).max():n}'
    return a2


# %% The solvers agree

fine = dict(dt=ss.days(0.25), total_pop=10_000) # Agents and stepped solvers differ from the ODE by an amount of order dt, so compare at a fine step; and at a size that runs quickly event by event
skip_solver = dict( # Corpus models that a solver cannot run, with the reason
    screening = dict(gillespie='rounds of a delivery need a step'),
)
loose = dict( # Wider tolerances, with the reason
    custom      = 0.10, # Few people are hospitalized, so the counts are noisy
    threshold   = 0.15, # The switch on and off at the threshold amplifies differences
    ages        = 0.10, # Each of 5,000 agents is 20 people, so the few who start infected make the timing of the epidemic noisy (3% with 50,000)
    age_vaccine = 0.12, # And a force of infection near 0.7 per day, so the difference of order dt is about 6% even at a quarter-day step (1% at a tenth)
    patches     = 0.10,
)


def mean_states(sim):
    """ The mean over time of each state, for each declared module """
    return {f'{mod.name}.{key}':res.values.mean() for mod in sim.declared for key,res in mod.results.items() if key.startswith('n_') and isinstance(res, ss.Result) and np.ndim(res.values) == 1}


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


@pytest.mark.parametrize('solver', ['ode', 'binomial', 'gillespie', 'tau', 'abm'])
def test_exit_order(solver):
    sc.heading(f'Testing that the order of competing exits does not matter under {solver}')
    shares = []
    for order in [('a', 'b', 'c'), ('c', 'a', 'b')]:
        I = ss.State(name='I')
        dests = {key:ss.State(name=key.upper()) for key in 'abc'}
        exits = {key:I >> dests[key] @ ss.perday(1.0) for key in order} # Three equal exits, fast compared with the step
        dis = ss.Disease(I, *dests.values(), **exits, name='exits')
        agents = dict(n_agents=20_000) if solver == 'abm' else {}
        pars = dict(solver='gillespie', total_pop=1e8) if solver == 'tau' else dict(solver=solver, total_pop=100_000) # Tau-leaping is gillespie's way with a run of too many events
        sim = ss.Sim(dis, unit='day', stop=5, verbose=0, **pars, **agents).run()
        assert sim.solver == solver
        res = sim.results.exits
        left = np.array([res[f'n_{k.upper()}'][-1] for k in 'abc'])
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
    )
    def run(form, solver):
        I, R, D = ss.states('I R D')
        dis = ss.Disease(I, R, D, **forms[form](I, R, D), name='cohort')
        size = dict(n_agents=20_000) if solver == 'abm' else dict(total_pop=20_000)
        res = ss.Sim(dis, unit='day', dt=0.05, stop=15, solver=solver, rand_seed=1, verbose=0, **size).run().results.cohort
        return np.array([res.n_I.values.mean(), res.n_R[-1], res.n_D[-1]])/20_000 # Time infected, and where people end up
    for form in forms:
        ode, abm = run(form, 'ode'), run(form, 'abm')
        assert np.allclose(abm, ode, rtol=0.03, atol=0.005), f'{form}: agents {abm}, ODE {ode}'
    return


def test_destinations():
    sc.heading('Testing that every solver agrees on the flow to each destination, by age')
    def run(solver, seed=1):
        I, R, D = ss.states('I R D')
        dis = ss.Disease(I, R, D, out=ss.Transition(I, [R, D], p=[0.7, 0.3], dur=ss.days(4)), init=dict(I=1000), name='cohort')
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
            total = sum(mod.results[f'n_{state}'].values for state in people_states(sim, mod))
            assert np.allclose(total, total[0], rtol=1e-9), f'{name} under {solver}: the population of {mod.name} goes from {total[0]:n} to {total[-1]:n}'
    return


def test_identities():
    sc.heading('Testing constant-rate decay and a fixed branch share against their exact values')

    # Constant-rate decay is exponential
    decay = ss.Disease(recovery=ss.progression('I -> R', rate=ss.perday(0.1)), name='decay')
    res = ss.Sim(decay, total_pop=1e6, unit='day', stop=50, solver='ode', verbose=0).run().results.decay
    t = np.arange(51)
    assert np.allclose(res.n_I.values, 1e6*np.exp(-0.1*t), rtol=1e-5) # RK4's error is about 1e-7 a step at this rate

    # A fixed branch share is the share that leaves by each branch
    for solver in ['ode', 'binomial', 'abm']:
        branch = ss.Disease(course=ss.progression('I -> R | D', dur=5, p=[None, 0.2]), name='branch')
        agents = dict(n_agents=50_000) if solver == 'abm' else {}
        res = ss.Sim(branch, total_pop=1e6, unit='day', stop=100, solver=solver, verbose=0, **agents).run().results.branch
        share = res.n_D[-1]/(res.n_R[-1] + res.n_D[-1])
        atol = 1e-9 if solver == 'ode' else 0.01 # Exact for the ODE; for a stochastic run, many standard errors of the share
        assert np.isclose(share, 0.2, atol=atol), f'{solver}: a share of {share:n} left by D, not 0.2'

    # A fixed delay is a conveyor: everyone who starts in E leaves exactly 10 days later, under every solver and as agents
    E, I = ss.states(2)
    for solver in ['ode', 'binomial', 'gillespie', 'abm']:
        size = dict(n_agents=1000) if solver == 'abm' else dict(total_pop=1000)
        cohort = ss.Disease(onset=E >> I @ ss.constant(ss.days(10)), name='cohort')
        n_E = ss.Sim(cohort, unit='day', stop=20, solver=solver, verbose=0, **size).run().results.cohort.n_E.values
        assert n_E[10] == 1000 and n_E[11] == 0, f'{solver}: {n_E[10]:n} in E on day 10 and {n_E[11]:n} on day 11, not 1000 and 0'

    # With a constant inflow as well, E holds the inflow over the delay: exactly in the ODE at every step, and on average in the stochastic solvers
    flow = ss.Disease(arrival=ss.null >> E @ ss.perday(100), onset=E >> I @ ss.constant(ss.days(10)), name='flow')
    exact = np.r_[1000 + 100*np.arange(11), np.full(20, 1000)] # Those who started in E leave on day 10, and the arrivals of each day 10 days later
    n_E = ss.Sim(flow, total_pop=1000, unit='day', stop=30, verbose=0).run().results.flow.n_E.values
    assert np.allclose(n_E, exact, rtol=1e-9)
    for solver in ['binomial', 'gillespie']:
        runs = ss.Sim(flow, total_pop=1000, unit='day', stop=30, solver=solver, verbose=0).run(n_runs=20).results.flow.n_E.values
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
    assert np.isclose(sim.diseases.malaria.R0(), 7.5) # √(abM/(Nr) · acσ/((σ + μ)μ)) with M = 50,000 mosquitoes at equilibrium and N = 10,000 people; was 0
    return


if __name__ == '__main__':
    T = sc.timer()
    for row in guesses:
        try:
            test_guesses_marked(row)
        except AssertionError as e:
            print(f'{row}: {e}')
    for name in corpus.compartmental:
        test_solvers_agree(name)
    T.toc()
