"""
Test that every set of kernels gives the same numbers: Numba's, Rust's and plain Python's bit for bit, and JAX's ODE to rounding

Numba's kernels are the reference: a change is made there first (plain Python is the same code, run uncompiled), then copied
to Rust's in starsim/solvers/rust/src/ and, where it applies, to JAX's in ss.gradients and ss.solvers.backends.chain (see starsim/solvers/rust/README.md). These are
quick checks of the main paths; tests/devtests/rust_twin.py compares every call to the compartmental kernels in any test.
Rust's are skipped if they have not been built (cd starsim/solvers/rust && ./build), unless STARSIM_RUST_REQUIRED=1 (as in CI).
"""

# %% Imports and settings
from unittest import mock
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
    return dict(solver='ode', integrator=solver) if solver in ['rk4', 'euler'] else dict(solver=solver)


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


# %% Define the tests

@sc.timer()
@needs_rust
def test_agents():
    sc.heading('Testing sims with agents: the passes, networks, demographics and draws of ss.Dist')
    same(lambda s: ss.Sim(n_agents=3000, diseases=ss.SIS(beta=0.1, dur_inf=ss.lognorm_ex(ss.years(1), ss.years(0.5))), networks='random', demographics=True, dur=20, rand_seed=s, **kw), compiled)
    same(lambda s: ss.Sim(n_agents=3000, diseases=ss.SEIR(), networks=['random', 'sexual'], dur=20, rand_seed=s, **kw), compiled)
    same(lambda s: ss.Sim(n_agents=2000, demographics=[ss.Pregnancy(), ss.Deaths()], networks='maternal', dur=10, rand_seed=s, **kw), compiled)
    return


@sc.timer()
def test_compartmental():
    sc.heading('Testing compartmental runs, compiled and as plain Python: every solver, with groups and places, whole and stepped')
    solvers = ['rk4', 'euler', 'binomial', 'tau', 'gillespie']
    for solver in solvers: # Small enough for plain Python
        same(lambda s: ss.Sim(make_seir(), total_pop=2000, unit='day', stop=60, rand_seed=s, **how(solver), **kw), compiled + ['python'], seeds=[2])
        same(lambda s: ss.Sim(make_seir(), total_pop=2000, unit='day', stop=40, rand_seed=s, branching=True, **how(solver), **kw), compiled + ['python'], seeds=[2]) # A branching process: the count in a flow's own group (ss.kernels.GROUP)
    if rust:
        for key in ['sir', 'commute', 'delay']:
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
def test_jax():
    sc.heading('Testing JAX: the ODE against the kernels, and the chain binomial with each set of compiled draws')
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
    if rust:
        make, pars = models['seir']
        with mock.patch.object(ss.solvers.choose, 'jax_after', 0): # Every run in JAX
            for solver in ['binomial', 'tau']:
                same(lambda s: ss.Sim(make(), **sc.mergedicts(pars, kw), solver=solver, rand_seed=s), compiled, seeds=[0])
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
    T.toc()
