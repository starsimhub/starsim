"""
Test branching processes (ss.Sim(branching=True)) under every solver, stopping a run with until=, and the 'ide' solver
"""

# %% Imports and settings
import pytest
import numpy as np
import scipy.optimize as spo
import sciris as sc
import starsim as ss

kw = dict(verbose=0)


def make_sir(n=1, beta=0.3, dur=10):
    """ An SIR with R0 = beta*dur """
    S, I, R = ss.states('S I R')
    return ss.Disease(S >> I @ ss.perday(beta)*I, I >> R @ ss.days(dur), init_n=dict(I=n), name='sir')


@sc.timer()
def test_named_branching():
    """ A module named like a setting of the sim leaves the setting alone: it isn't a branching process """
    sc.heading('Testing a module named "branching"')
    S, I, R = ss.states(3)
    sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(8), init=dict(I=0.01), name='branching')
    with pytest.warns(RuntimeWarning, match='has an attribute'):
        sim = ss.Sim(sir, unit='day', stop=100, total_pop=1000, verbose=0).run()
    assert sim.pars.branching is False and sim.diseases.branching.results.S[-1] < 500
    return


def test_held():
    sc.heading('Infection takes no one from the uninfected under every solver, whose own flows act as usual')
    for solver in ['ode', 'binomial', 'gillespie', 'tau', 'abm']:
        sim = ss.Sim(make_sir(n=10), total_pop=1e6, branching=True, solver=solver, unit='day', stop=30, **kw).run()
        res = sim.results.sir
        assert np.all(res.S.values == 1e6 - 10), solver
        assert res.I[-1] > 10, solver
    sim = ss.Sim(make_sir(n=10), total_pop=1e6, branching=True, solver='ode', unit='day', stop=60, **kw).run() # dI/dt = (beta - gamma)*I
    assert np.isclose(sim.results.sir.I[-1], 10*np.exp(0.2*60), rtol=1e-3)
    assert ss.Sim(make_sir(), total_pop=1e6, branching=True, solver='abm', unit='day', stop=10, **kw).init().people.n_uids == 1 # Agents only for the infected
    for solver in ['ode', 'abm']: # Vaccination acts on the uninfected, and waning back to them takes people out of the process
        S, V, I, R = ss.states(4)
        dis = ss.Disease(S >> I @ ss.perday(0.3)*I, S >> V @ ss.perday(0.02), I >> R @ ss.days(10), R >> S @ ss.days(30), init_n=dict(I=20), name='sirs')
        sim = ss.Sim(dis, total_pop=1e6, branching=True, solver=solver, unit='day', stop=60, verbose=0)
        sim.run()
        assert np.isclose(sim.results.sirs.S[-1], (1e6 - 20)*np.exp(-0.02*60), rtol=1e-3), solver
        assert any('go back to the uninfected' in note for note in sim.notes), solver
    return sim


@sc.timer()
def test_until():
    sc.heading('A run stopped by a condition: an expression of the model, or a function of the sim')
    sir = make_sir(n=10)
    sim = ss.Sim(sir, total_pop=1e6, branching=True, solver='gillespie', unit='day', stop=200, **kw)
    sim.run(until=sir.I.n + sir.R.n >= 500)
    I, R = sim.results.sir.I, sim.results.sir.R
    assert sim.complete and sim.stopped is not None and sim.branching.outcome == 'stopped'
    assert I[sim.stopped] + R[sim.stopped] >= 500 and I[sim.stopped - 1] + R[sim.stopped - 1] < 500
    assert np.isnan(I[sim.stopped + 1:]).all()
    sim = ss.Sim(make_sir(n=10), n_agents=2000, unit='day', stop=200, **kw) # Not only for branching: an ordinary run, by a function of the sim
    sim.run(until=lambda sim: sim.t.ti >= 20)
    assert sim.stopped == 20
    return sim


@sc.timer()
def test_extinction():
    sc.heading('The chance a single case dies out is 1/R0, under the exact solver and as agents')
    for solver in ['gillespie', 'abm']:
        sir = make_sir()
        sim = ss.Sim(sir, total_pop=1e6, branching=True, solver=solver, unit='day', stop=100, **kw)
        msim = ss.parallel(sim, n_runs=100, parallel=False, run_args=dict(until=sir.I.n + sir.R.n >= 50)) # In series: each run takes a few milliseconds
        b = msim.branching
        assert set(b.outcomes) == {'extinct', 'stopped'}, solver
        assert abs(b.p_extinct - 1/3) < 0.15, (solver, b.p_extinct) # About three standard errors
        assert b.R is None # Not fitted from runs stopped by until=, whose stop depends on their offspring
    return msim


@sc.timer()
def test_offspring():
    sc.heading('As agents: a gamma-distributed infectiousness gives negative binomial offspring, and each case has an infector and a generation')
    k = 0.5
    S, I, R = ss.states(3)
    dis = ss.Disease(S >> I @ ss.perday(0.15)*I, I >> R @ ss.constant(ss.days(8)), init_n=dict(I=300), rel_trans=ss.attr(ss.gamma(a=k, scale=1/k)), name='sir')
    sim = ss.Sim(dis, total_pop=1e7, branching=True, solver='abm', unit='day', stop=100, **kw).run()
    b = sim.branching
    R0, kfit = ss.run.Branching.fit_offspring(b.offspring[b.complete])
    assert abs(R0 - 1.2) < 0.08 and abs(kfit - k) < 0.12, (R0, kfit)
    seeds = b.infector == -1
    assert seeds.sum() == 300 and np.all(b.generation[seeds] == 0)
    child = np.flatnonzero(~seeds)[0]
    assert b.generation[child] == b.generation[b.infector[child]] + 1
    assert b.length == b.generation.max() > 3
    return sim


@sc.timer()
def test_ide():
    sc.heading('The ide solver: a dwell time as it is, and under branching, the renewal equation')
    def make(dur, solver, dt=1.0, branching=False, stop=150):
        S, E, I, R = ss.states(4)
        dis = ss.Disease(S >> E @ ss.perday(0.4)*I, E >> I @ dur, I >> R @ ss.perday(0.2), init_n=dict(I=10), name='seir')
        return ss.Sim(dis, total_pop=1e6, solver=solver, unit='day', dt=dt, stop=stop, branching=branching, **kw).run().results.seir
    gamma = ss.gamma(a=4, scale=ss.days(2)) # Exact as four Erlang stages
    ode, ide = make(gamma, 'ode', 0.25), make(gamma, 'ide', 0.25)
    assert np.allclose(ode.I.values, ide.I.values, rtol=2e-3, atol=1)
    delay = ss.constant(ss.days(6)) # Exact as a conveyor, as it already is
    assert np.array_equal(make(delay, 'ode').I.values, make(delay, 'ide').I.values)
    res = make(gamma, 'ide', branching=True, stop=100) # Grows at the rate the generation interval gives (Euler-Lotka)
    y = res.E.values + res.I.values
    t = np.arange(len(y))
    r = np.polyfit(t[60:], np.log(y[60:]), 1)[0]
    r_true = spo.brentq(lambda r: 0.4*(1 + 2*r)**-4/(r + 0.2) - 1, 1e-6, 2)
    assert abs(r - r_true) < 1e-3, (r, r_true)
    return res


if __name__ == '__main__':
    sc.options(interactive=False)
    T = sc.timer()
    test_held()
    test_until()
    test_extinction()
    test_offspring()
    test_ide()
    T.toc()
