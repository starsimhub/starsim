"""
Test that every set of kernels gives the same numbers: Numba's, Rust's and plain Python's bit for bit, and JAX's ODE to rounding

Numba's kernels are the reference: a change is made there first (plain Python is the same code, run uncompiled), then copied
to Rust's in starsim/solvers/rust/src/ and, where it applies, to JAX's in ss.gradients (see starsim/solvers/rust/README.md). These are
quick checks of the main paths, plus one direct call of every kernel that Rust's replace (test_twins()), and a check that with Rust's on no kernel of Numba's runs (test_no_numba()), so a kernel written in Numba alone fails; tests/devtests/rust_twin.py compares every call to the compartmental kernels in any test.
Rust's are skipped if they have not been built (cd starsim/solvers/rust && ./build), unless STARSIM_RUST_REQUIRED=1 (as in CI).
"""

# %% Imports and settings
import os
import sys
import importlib
import subprocess
from unittest import mock
import numba as nb
import pytest
import numpy as np
import sciris as sc
import starsim as ss
from test_kernels import models, make_seir

kw = dict(verbose=0)
D = ss.solvers.backends.draws
rust = ss.solvers.backends.rust.built() or sc.parse_env('STARSIM_RUST_REQUIRED', False, bool) # Whether to compare Rust's
needs_rust = pytest.mark.skipif(not rust, reason='the Rust kernels have not been built (cd starsim/solvers/rust && ./build)')
compiled = ['numba', 'rust'] if rust else ['numba']


def snapshot(sim):
    """ Every result (and its strata), and for agents every state and network edge """
    out = {}
    for k, v in sim.results.flatten().items():
        if isinstance(v, (np.ndarray, ss.Result)):
            out[k] = np.asarray(v)
            if getattr(v, 'strata', None) is not None:
                out[k + '.strata'] = np.asarray(v.strata)
    if getattr(sim, 'people', None) is not None and hasattr(sim.people, 'states'):
        out.update({f'people.{k}': np.asarray(v.raw) for k, v in sim.people.states.items()})
        for name, net in sim.networks.items():
            if getattr(net, 'edges', None) is not None:
                out.update({f'{name}.{k}': np.asarray(v) for k, v in net.edges.items()})
    return out


def how(solver):
    """ The sim's arguments for a solver: rk4 and euler are integrators of the ODE """
    return dict(solver=ss.ode(integrator=solver)) if solver in ['rk4', 'euler'] else dict(solver=solver)


def same(make, kernels, seeds=(0, 1)):
    """ Run each sim with each set of kernels; check they all give the same numbers, with no call going from Rust's to Numba's """
    ss.solvers.backends.rust.fell.clear()
    for seed in seeds:
        res = {}
        for k in kernels:
            with ss.options.context(kernels=k):
                res[k] = snapshot(make(seed).run())
        ref = res[kernels[0]]
        for k in kernels[1:]:
            assert res[k].keys() == ref.keys()
            differ = [key for key in ref if not np.array_equal(ref[key], res[k][key], equal_nan=True)]
            assert not differ, f'{k} differs from {kernels[0]} (seed {seed}): {differ[:5]}'
    assert not ss.solvers.backends.rust.fell, f'Calls that went from Rust to Numba: {dict(ss.solvers.backends.rust.fell)}'
    return


event_calls = [] # The arguments of the calls of the events' pass in two sims, once taken (see events_inputs())

def events_inputs():
    """
    The arguments of every call of the pass of solver=ss.abm(step='events') in two small sims, kept the first time: one on a network with acts (by
    direction) and a mixing pool, with an Erlang dwell, a dwell drawn for each agent, a split with deaths, and waning; and one by age, with a contact matrix
    """
    if not event_calls:
        S, E, I, R = ss.states('S E I R')
        d = ss.Disease(infection=S >> E @ dict(sexualnet=ss.peract(m2f=0.2, f2m=0.1), mixingpool=ss.perday(0.02))*I, latent=ss.Transition(E, I, dur=ss.erlang(k=3, mean=ss.days(5))),
                       recovery=ss.Transition(I, {R: 0.9, ss.dead: 0.1}, dur=ss.lognormal(mean=ss.days(8), std=ss.days(3))), waning=R >> S @ ss.perday(0.05), init_prev=0.05, name='d')
        acts = ss.Sim(d, networks=[ss.SexualNet(acts=ss.constant(ss.per(80, 'year'))), ss.MixingPool(n_contacts=3)], n_agents=1000, solver=ss.abm(step='events'), unit='day', stop=30, **kw)
        S, I, R = ss.states('S I R')
        m = ss.Disease(S >> I @ 0.14*I, I >> R @ ss.days(8), init_n=dict(I=10), name='m')
        m.stratify(age={0: 0.3, 15: 0.6, 65: 0.1})
        ages = ss.Sim(m, networks=ss.contacts(ages=[0, 15, 65], matrix=[[10, 6, 1], [3, 8, 1], [3, 6, 3]]), n_agents=1000, solver=ss.abm(step='events'), unit='day', stop=30, **kw)
        E = ss.solvers.backends.events
        step_events = E.step_events
        def keep(*args):
            event_calls.append([np.array(x) if isinstance(x, np.ndarray) else x for x in args]) # Copies, since the pass changes the agents' arrays
            return step_events(*args)
        with mock.patch.object(E, 'step_events', keep):
            acts.run()
            ages.run()
    return event_calls


def events_call(args):
    """ A call of the events' pass on copies of its arguments: what it returns, and the agents' arrays it changes (state, ver, t_exit, exit_tr) """
    args = [np.array(x) if isinstance(x, np.ndarray) else x for x in args]
    return ss.solvers.backends.events.step_events(*args), args[3:7]


move_calls = [] # The arguments of the calls of the compiled moves in a sim, once taken (see moves_inputs())

def moves_inputs():
    """
    The arguments of every call of the compiled moves (`due_agents()` and `step_moves()`) in two small sims, kept the first time: a lognormal
    dwell into a split, a split with ss.stay a time after an event, a split with deaths, and a competing rate; and dwell times that never end
    outside a where= on another module, a split whose shares are worked out as they are reached, and a trigger on a condition
    """
    if not move_calls:
        S, E, I, R = ss.states('S E I R')
        onset = ss.Transition(E, {I: 0.7, R: None}, dur=ss.lognormal(mean=ss.days(3), std=ss.days(1)))
        d = ss.Disease(infection=S >> E @ 0.1*I, onset=onset, care=ss.Transition(I, {I: 0.5, ss.stay: None}, when=ss.on(onset, delay=ss.days(4))),
                       recovery=ss.Transition(I, {R: 0.95, ss.dead: None}, dur=ss.weibull(shape=2, scale=ss.days(8))), waning=R >> S @ ss.perday(0.05), init_prev=0.05, name='d')
        sim = ss.Sim(d, networks=ss.RandomNet(n_contacts=4), n_agents=1000, unit='day', stop=30, **kw)
        M = ss.solvers.backends.moves
        kernels = {k: getattr(M, k) for k in ['due_agents', 'step_moves']}
        def keeper(k):
            def keep(*args):
                move_calls.append((k, copied(args)))
                return kernels[k](*args)
            return keep
        A, B = ss.states('A B')
        X, Y, Z = ss.states('X Y Z')
        clock = ss.Disease(A, B, tick=A >> B @ ss.constant(ss.days(5)), default_state='A', name='clock')
        share = lambda sim, uids: np.full(len(uids), 0.3)
        dec = ss.Disease(X, Y, Z, to_y=ss.Transition(X, Y, p=0.5, rate=ss.perday(2), who=clock.B), to_z=ss.Transition(X, Z, p=0.5, rate=ss.perday(2), who=clock.B),
                         onward=ss.Transition(Y, {Z: share, ss.stay: None}, dur=ss.days(2)), back=ss.Transition(Z, X, dur=0, who=ss.person.female), default_state='X', name='dec')
        sim2 = ss.Sim([clock, dec], n_agents=1000, unit='day', stop=20, **kw)
        with mock.patch.object(M, 'due_agents', keeper('due_agents')), mock.patch.object(M, 'step_moves', keeper('step_moves')):
            sim.run()
            sim2.run()
        eng = sim2.diseases.dec.engine
        assert not eng.uncompiled and eng.kplan.lazy.any() and len(eng.kplan.cond_trigs), eng.uncompiled # Each of them taken by the call
        assert {k for k,_ in move_calls} == {'due_agents', 'step_moves'}
    return move_calls


def copied(args):
    """ Copies of the arrays among some arguments (and in their tuples), since the moves change them """
    return [tuple(np.array(y) for y in x) if isinstance(x, tuple) else np.array(x) if isinstance(x, np.ndarray) else x for x in args]


def moves_call(k, args):
    """ A call of the compiled moves on copies of its arguments: what it returns, and every array it was given, which it changes in place """
    args = copied(args)
    return getattr(ss.solvers.backends.moves, k)(*args), [x for x in args if isinstance(x, (np.ndarray, tuple))]


def twin_calls():
    """ A call of each kernel in `rust.names`, through the function that calls it where there is one, each giving what it returns and any array it fills """
    P, D, net, rng = ss.solvers.backends.passes, ss.solvers.backends.draws, ss.networks, np.random.default_rng(1)
    n = 1000
    uni = lambda dtype=float, size=n: rng.random(size).astype(dtype)
    auids, gone = np.arange(0, n, 2), np.arange(0, n, 50)
    alive = uni() < 0.9
    a, b = alive & (uni() < 0.5), alive & (uni() < 0.2) # In a state, so alive
    groups, ages, debut = rng.integers(0, 3, n).astype(np.int32), 100*uni(np.float32), 20*uni(np.float32)
    ti_dead, ti_removed = 10*uni(np.float32), 20*uni(np.float32)
    rate, adjust, sus = uni(size=3), uni(size=3), uni(np.float32)
    p1, p2, trans, randvals = rng.integers(0, n, 3000), rng.integers(0, n, 3000), uni(np.float32)*a, uni(np.float32, 3000)
    dur = 3*uni(np.float32, 3000)
    index = lambda: P.edge_index(p1, p2, n) # The edges of each agent: s1, e1, o1, s2, e2, o2
    def filled(f, *args, size=n, dtype=np.int64):
        out = np.empty(size, dtype=dtype)
        return f(*args, out), out
    def changed(f, x, *args):
        x = x.copy()
        return f(x, *args), x
    return dict(
        levels      = lambda: P.levels(3*uni()),
        crossed     = lambda: P.crossed(uni(), groups, np.arange(3.0), np.arange(3.0) + 0.5),
        probs       = lambda: P.infection_probs(a, auids, groups, rate, sus, 0.8, adjust, 1.0, 0.1),
        census      = lambda: P.census((a, b), alive, groups, 3, np.array([0, -1])),
        nonzero     = lambda: [P.flatnonzero(a), P.flatnonzero_serial(a, 1)],
        infect      = lambda: P.infect(a, groups, rate, sus, 0.8, adjust, 1.0, 0.1, None, 1, 2, np.float32),
        due         = lambda: P.due(auids, ti_dead, ti_removed, 5),
        keep        = lambda: changed(lambda age: (P.keep_alive(auids, alive, age, np.float32(0.1), True, np.empty(n, dtype=np.int64)), P.keep_raw(alive, age, np.float32(0.1), True, np.empty(n, dtype=np.int64), int(alive.sum()))), ages),
        due_alive   = lambda: P.due_alive(alive, ti_dead, ti_removed, 5),
        alive       = lambda: P.count_alive((a, b), alive),
        edges       = lambda: [P.random_edges(auids, 3, 1, True), P.random_edges(auids, rng.integers(0, 4, len(auids)), 1, False)],
        candidates  = lambda: P.candidates(p1, p2, trans, sus),
        indexed     = lambda: [index(), P.partners(gone, *[index()[i] for i in [0, 2, 3, 5]], n)] + [P.candidates_indexed(trans, sus, two_way, *index(), most) for two_way in [True, False] for most in [0, 3000]],
        can         = lambda: P.can_transmit(p1, p2, trans, sus, np.float32(0.5)),
        ending      = lambda: P.ending(dur, p1, p2, alive),
        touching    = lambda: P.touching(p1, p2, b),
        women       = lambda: P.women(auids, a, ages, 15, 50),
        debuts      = lambda: P.debuts(auids, alive, a, ages, debut, True),
        compress    = lambda: P.compress(uni(), gone),
        exits       = lambda: P.competing_exits(auids, None, 1, 2, np.float32, 0.3, np.array([0.1, 0.3])),
        agent_exits = lambda: P.competing_agent_exits(auids, None, 1, 2, np.float32, uni(size=len(auids)), uni(size=len(auids)), uni(size=(2, len(auids)))),
        true        = lambda: [ss.arrays.nb_true_raw(auids, a, True), ss.arrays.nb_true(auids, a[auids], False), ss.arrays.nb_count(auids, a), ss.arrays.nb_indexer(ages, auids)],
        transmit    = lambda: net.nb_transmit(p1, p2, trans, sus, randvals, uni(np.float32, 3000)),
        shuffle     = lambda: changed(net.fisher_yates_shuffle, auids, uni(size=len(auids))),
        partners    = lambda: changed(lambda x: net._mark_partners(p1, p2, a, x), np.zeros(n, dtype=bool)),
        hashes      = lambda: ss.dists.hash_uniforms(1, 2, auids, np.float32),
        combine     = lambda: [ss.multi_random.combine2_rvs(uni(np.float32), uni(np.float32), np.uint32, 2**32 - 1), ss.multi_random.combine_rvs(nb.typed.List([uni(np.float32) for _ in range(3)]), np.uint32, 2**32 - 1)],
        events      = lambda: [events_call(args) for args in events_inputs()],
        moves       = lambda: [moves_call(k, args) for k,args in moves_inputs()],
        many        = lambda: [filled(D.many[k], *pars, uni(), dtype=float) for k,pars in [('expon', [uni()]), ('weibull', [uni(), uni()]), ('binomial', [10*uni(), uni()]), ('poisson', [5*uni()]), ('nbinom', [uni() + 1, uni()])]],
    )


def flat(x):
    """ The arrays and numbers in what a kernel gave, in order """
    if x is None:
        return []
    return [v for item in x for v in flat(item)] if isinstance(x, (tuple, list)) else [np.asarray(x)]


@sc.timer()
@needs_rust
def test_twins():
    sc.heading('Testing every kernel that Rust replaces against Numba\'s, directly: serial and threaded, with no call going back to Numba')
    R = ss.solvers.backends.rust
    mods = R.modules()
    hit = set()
    def counted(key, f):
        def call(*args):
            hit.add(key)
            return f(*args)
        return call
    res = {}
    R.fell.clear()
    for kernels in ['numba', 'rust']:
        for chunks in [1, 3]: # Serial and threaded, whatever the size
            with ss.options.context(kernels=kernels), mock.patch.object(ss.solvers.backends.passes, 'n_chunks', lambda n, width=1: chunks):
                for m,keys in R.names.items():
                    for k in keys:
                        f = R.get(mods[m], k)
                        pair = isinstance(f, (list, tuple))
                        R.put(mods[m], k, type(f)(counted((m, k, i), g) for i,g in enumerate(f)) if pair else counted((m, k), getattr(f, '__func__', f)))
                res[kernels, chunks] = {k:flat(f()) for k,f in twin_calls().items()}
    ref = res['numba', 1]
    for key,got in res.items():
        differ = [k for k in ref if not all(np.array_equal(u, v, equal_nan=True) for u,v in zip(ref[k], got[k], strict=True))]
        assert not differ, f'{key} differs from Numba\'s serial kernels: {differ}'
    assert not R.fell, f'Calls that went from Rust to Numba: {dict(R.fell)}'
    drivers = {(m, k, i) for m,keys in R.names.items() for k in keys for i in range(2) if k.endswith('_drivers')}
    want = {(m, k) for m,keys in R.names.items() for k in keys if not k.endswith('_drivers')} | drivers
    assert not want - hit, f'Kernels not called: {sorted(want - hit)}'
    return


@sc.timer()
def test_stale():
    sc.heading('Testing that Rust kernels built from other source than is here are not used silently')
    R = ss.solvers.backends.rust
    pytest.importorskip('starsim.solvers.backends._rs')
    with mock.patch.object(R, 'source_hash', lambda: 0):
        with pytest.raises(AttributeError, match='rebuild'):
            R.use(True)
    if rust and R.source_hash() is not None: # None when run against a wheel, which has no source beside it
        assert R.source_hash() == R.load().source_hash, 'Rebuild the Rust kernels'
    return


@sc.timer()
@needs_rust
def test_no_numba():
    sc.heading('Testing that with the Rust kernels no kernel of Numba\'s runs: every one that runs a sim has a Rust copy')
    script = """
import importlib, numba, starsim as ss, starsim.library as ssl
ss.options(kernels='rust')
kw = dict(verbose=0)
S, I, R = ss.states('S I R')
sir = lambda: ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init_prev=0.01, name='sir')
sims = [ss.Sim(sir(), networks=net, n_agents=1000, unit='day', stop=30, solver=solver, **kw) for net,solver in [(ss.RandomNet(), 'abm'), (ssl.StaticNet(n_contacts=10), 'abm'), (ssl.StaticNet(n_contacts=10), ss.abm(step='events')), (ss.MixingPool(n_contacts=ss.constant(5)), ss.abm(step='events'))]]
sims += [ss.Sim(ss.SEIR(beta=dict(randomnet=ss.peryear(0.1), sexualnet=ss.peract(0.1))), networks=['random', 'sexual'], demographics=True, n_agents=1000, stop=ss.years(5), **kw)]
sims += [ss.Sim(sir(), total_pop=10000, unit='day', stop=30, solver=solver, **kw) for solver in ['ode', 'binomial', 'gillespie', 'tau']]
for sim in sims:
    sim.run()
used = [f'{name}.{k}' for name in ['passes', 'events', 'kernels', 'draws'] for mod in [importlib.import_module(f'starsim.solvers.backends.{name}')] for k,f in vars(mod).items() if isinstance(f, numba.core.registry.CPUDispatcher) and f.__module__ == mod.__name__ and f.signatures]
print(used)
"""
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True, env=os.environ | dict(MPLBACKEND='agg')) # A fresh process, so no other test's kernels have been compiled
    used = out.stdout.strip().splitlines()[-1]
    assert used == '[]', f'These Numba kernels ran with the Rust kernels on, so have no Rust copy (or one that is not used): {used}; copy them to starsim/solvers/rust/src/ and add them to rust.names'
    return


@sc.timer()
def test_stale_numba(tmp_path):
    sc.heading('Testing that Numba\'s cache is checked against the contents of a file, not its time')
    C = ss.solvers.backends.cache
    path = tmp_path/'kernels.py'
    path.write_text('x = 1')
    before = C.contents(str(path))
    os.utime(path, (1, 1)) # As a fresh checkout does
    assert C.contents(str(path)) == before, 'A new time should keep a cache that is still right'
    path.write_text('x = 2') # The same size
    os.utime(path, (1, 1))
    C.hashes.clear() # As a new process does
    assert C.contents(str(path)) != before, 'A change with the same time and size should make a cache stale'
    return


# %% Define the tests

@sc.timer()
@needs_rust
def test_agents():
    sc.heading('Testing sims with agents: the passes, networks, demographics and draws of ss.Dist, and agents by events')
    same(lambda s: ss.Sim(n_agents=3000, diseases=ss.SIS(beta=0.1, dur_inf=ss.lognormal(ss.years(1), ss.years(0.5))), networks='random', demographics=True, stop=ss.years(20), rand_seed=s, **kw), compiled)
    same(lambda s: ss.Sim(n_agents=3000, diseases=ss.SEIR(beta=dict(randomnet=ss.peryear(0.1), sexualnet=ss.peract(0.1))), networks=['random', 'sexual'], stop=ss.years(20), rand_seed=s, **kw), compiled)
    same(lambda s: ss.Sim(n_agents=2000, demographics=[ss.Pregnancy(), ss.Deaths()], networks='prenatal', stop=ss.years(10), rand_seed=s, **kw), compiled)
    S, I, R = ss.states('S I R') # Agents by events (the pass is also called directly in test_twins())
    sirs = lambda: ss.Disease(infection=S >> I @ 0.05*I, recovery=I >> R @ 1, waning=R >> S @ (1/14), init_prev=0.01, name='sirs')
    same(lambda s: ss.Sim(sirs(), networks=[ss.RandomNet(n_contacts=10), ss.MixingPool(n_contacts=2)], n_agents=2000, solver=ss.abm(step='events'), unit='day', stop=30, rand_seed=s, **kw), compiled)
    return


@sc.timer()
def test_compartmental():
    sc.heading('Testing compartmental runs, compiled and as plain Python: every solver, with groups and places, whole and stepped')
    solvers = ['rk4', 'euler', 'binomial', 'tau', 'gillespie']
    for solver in solvers: # Small enough for plain Python
        same(lambda s: ss.Sim(make_seir(), total_pop=2000, unit='day', stop=60, rand_seed=s, **how(solver), **kw), compiled + ['python'], seeds=[2])
        same(lambda s: ss.Sim(make_seir(), total_pop=2000, unit='day', stop=40, rand_seed=s, branching=True, **how(solver), **kw), compiled + ['python'], seeds=[2]) # A branching process: the count in a flow's own group (ss.kernels.GROUP)
    if rust:
        for key in ['sir', 'commute', 'delay', 'biting']:
            make, pars = models[key]
            for solver in solvers:
                if solver == 'gillespie' and key == 'commute':
                    continue # Too slow for a quick test
                same(lambda s: ss.Sim(make(), **sc.mergedicts(pars, kw), **how(solver), rand_seed=s), compiled, seeds=[0])
        make, pars = models['towns'] # A branching process by place, which the exact solver steps place by place
        for solver in ['binomial', 'gillespie']:
            same(lambda s: ss.Sim(make(), **sc.mergedicts(pars, kw, dict(stop=30)), **how(solver), branching=True, rand_seed=s), compiled, seeds=[0])
        make, pars = models['seir']
        def stepped(seed):
            sim = ss.Sim(make(), **sc.mergedicts(pars, kw), solver='binomial', rand_seed=seed)
            sim.run(until=50)
            return sim
        same(stepped, compiled, seeds=[0])
    return


@sc.timer()
def test_full_share():
    sc.heading('Testing a time share with an entry for every pair of places: by BLAS, against each set of kernels and the entries one at a time')
    rng = np.random.default_rng(1)
    n, levels = 20, 2
    pop, lat, lon = rng.lognormal(10, 0.5, n), rng.uniform(5, 11, n), rng.uniform(-3, 1, n)
    def make(seed):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(S >> I @ ss.perday(0.5) * I, I >> R @ ss.days(8), init_prev=0.01, name='sir')
        sir.stratify(place=ss.levels(ss.geo.geography(names=[f'p{i}' for i in range(n)], populations=pop, lat=lat, lon=lon), mobility=ss.geo.gravity(distance_power=1.5, kind='commute', away=0.05))).stratify(age={0: 0.3, 15: 0.7})
        return ss.Sim(sir, solver='binomial', total_pop=pop.sum(), unit='day', stop=30, rand_seed=seed, verbose=0)
    same(make, compiled)
    pairs = np.array([np.repeat(np.arange(n), n), np.tile(np.arange(n), n)])
    vals, v, out = rng.random(n*n), rng.random(n*levels), np.empty(n*levels)
    matrix = vals.reshape(n, n)
    for by_columns in [True, False]:
        ss.kernels.share(pairs, vals, v, n*levels, levels, out, by_columns)
        ref = (v.reshape(n, levels).T @ matrix).T if by_columns else matrix @ v.reshape(n, levels)
        assert np.allclose(out, ref.ravel(), rtol=1e-12)
    return


@sc.timer()
def test_met_counts():
    sc.heading('Testing counts that a contact matrix acts on (beta*I.n, beta*I.frac, and weighted), by age within towns with commuting: each set of kernels, and JAX')
    def make(seed=0, weighted=False, **kwargs):
        towns = ss.geo.geography(names=['north', 'south', 'east'], populations=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
        S, I, R = ss.states('S I R')
        I.rel_trans = 0.5
        rate = 1e-6*(ss.by(age=[2, 1, 0.5])*I).n + 0.2*I.n_infectious/(S + I + R).n if weighted else 2e-6*I.n + 0.1*I.frac
        sir = ss.Disease(infection=S >> I @ rate, recovery=I >> R @ ss.days(8), init_n=dict(I=ss.by(patch=[50, 0, 0])), name='sir')
        sir.stratify(patch=ss.levels(towns, mobility=ss.geo.gravity(scale=2e-4))).stratify(age={0: 0.3, 15: 0.5, 65: 0.2})
        return ss.Sim(sir, networks=ss.contacts(np.array([[8, 3, 1], [3, 6, 2], [1, 2, 3]])), unit='day', stop=100, total_pop=100_000, rand_seed=seed, **kw, **kwargs)
    for weighted in [False, True]:
        for solver in ['rk4', 'binomial']:
            same(lambda s: make(s, weighted, **how(solver)), compiled + ['python'], seeds=[0])
        if importlib.util.find_spec('jax'):
            ode, ref = make(weighted=weighted, solver='ode'), make(weighted=weighted, solver='ode')
            ode.jax, ref.jax = True, False
            assert np.allclose(ode.run().results.sir.I.values, ref.run().results.sir.I.values, rtol=1e-9, atol=0)
    return


@sc.timer()
def test_jax():
    sc.heading('Testing JAX: the ODE against the kernels')
    pytest.importorskip('jax')
    make, pars = models['towns']
    ode = ss.Sim(make(), **sc.mergedicts(pars, kw), solver='ode')
    ode.jax = True
    ref = ss.Sim(make(), **sc.mergedicts(pars, kw), solver='ode')
    ref.jax = False
    assert np.allclose(ode.run().results.sir.I.values, ref.run().results.sir.I.values, rtol=1e-9, atol=0) # To rounding: XLA orders the arithmetic its own way
    ode = ss.Sim(make(), **sc.mergedicts(pars, kw), solver='ode', branching=True) # A branching process, by place
    ode.jax = True
    ref = ss.Sim(make(), **sc.mergedicts(pars, kw), solver='ode', branching=True)
    ref.jax = False
    assert np.allclose(ode.run().results.sir.I.values, ref.run().results.sir.I.values, rtol=1e-9, atol=0)
    return


@sc.timer()
def test_draws():
    sc.heading('Testing the exact draws at the edges of their ranges, compiled and as plain Python')
    P = D.plain_copy()
    u = np.random.default_rng(1).random(400)
    u[:4] = [2**-53, 0.5, 1 - 2**-53, 0.075]
    n = np.array([-1, 0, 1, 20, 100, 1000, 1e6, 1e17, np.inf, np.nan]).repeat(40)
    p = np.tile([0, 1e-300, 1e-6, 0.05, 0.3, 0.5, 0.7, 1 - 1e-12, 1, np.nan], 40)
    lam = np.tile([-1, 0, 0.01, 2, 29.9, 30.1, 1e6, 1e19, np.inf, np.nan], 40)
    with np.errstate(over='ignore', invalid='ignore'): # The plain copy's random streams overflow, on purpose (as compiled code does silently)
        a, b = np.nan_to_num(np.abs(lam[:200]) % 50, posinf=1) + 0.5, np.nan_to_num(np.abs(p[:200]) % 1) + 1e-3 # nbinom's parameters, valid
        ref = dict(
            binomial  = [D.binomial(*x) for x in zip(n, p, u)],
            poisson   = [D.poisson(*x) for x in zip(lam, u)],
            nbinom    = [D.nbinom(*x) for x in zip(a, b, u[:200])],
            binomials = D.binomials(n, p, u, np.empty(400), np.empty((9, 400))),
            poissons  = D.poissons(lam, u, np.empty(400), np.empty((6, 400))),
        )
        plain = dict(
            binomial  = [P.binomial(*x) for x in zip(n, p, u)],
            poisson   = [P.poisson(*x) for x in zip(lam, u)],
            nbinom    = [P.nbinom(*x) for x in zip(a, b, u[:200])],
            binomials = P.binomials(n, p, u, np.empty(400), np.empty((9, 400))),
            poissons  = P.poissons(lam, u, np.empty(400), np.empty((6, 400))),
        )
    sets = dict(python=plain)
    if rust:
        rs = ss.solvers.backends.rust.load()
        def into(f, *args, size=400):
            out = np.empty(size)
            f(*args, out)
            return out
        sets['rust'] = dict(
            binomial  = into(rs.draws_binomial, n, p, u),
            poisson   = into(rs.draws_poisson, lam, u),
            nbinom    = into(rs.draws_nbinom, a, b, u[:200], size=200),
            binomials = into(lambda *x: rs.draws_binomials(*x, np.empty((9, 400))), n, p, u),
            poissons  = into(lambda *x: rs.draws_poissons(*x, np.empty((6, 400))), lam, u),
        )
    for kernels, got in sets.items():
        for name in ref:
            assert np.array_equal(got[name], ref[name], equal_nan=True), f'{kernels}: {name}'
    return


if __name__ == '__main__':
    T = sc.timer()
    test_agents()
    test_compartmental()
    test_jax()
    test_draws()
    test_twins()
    test_no_numba()
    test_stale()
    T.toc()
