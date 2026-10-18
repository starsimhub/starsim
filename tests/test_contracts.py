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
        dis = ss.Disease(I, *dests.values(), **exits, init=dict(I=1.0), name='exits')
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
