"""
Test branching processes (ss.Sim(branching=True)) under every solver, stopping a run with until=, exact dwell times (ss.solvers.ode(dwells='exact')),
and the conveyor: fixed delays kept as levels of whole steps under the compartmental solvers; and clocks, which are agents only
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


def test_held():
    sc.heading('Infection takes no one from the uninfected under every solver, whose own flows act as usual')
    for solver in ['ode', 'binomial', 'gillespie', 'tau', 'abm']:
        res = ss.Sim(make_sir(n=10), total_pop=1e6, branching=True, solver=solver, unit='day', stop=30, **kw).run().results.sir
        assert np.all(res.S.values == 1e6 - 10) and res.I[-1] > 10, solver
    sim = ss.Sim(make_sir(n=10), total_pop=1e6, branching=True, solver='ode', unit='day', stop=60, **kw).run() # dI/dt = (beta - gamma)*I
    assert np.isclose(sim.results.sir.I[-1], 10*np.exp(0.2*60), rtol=1e-3)
    assert ss.Sim(make_sir(), total_pop=1e6, branching=True, solver='abm', unit='day', stop=10, **kw).init().people.n_uids == 1 # Agents only for the infected
    for solver in ['ode', 'abm']: # Vaccination acts on the uninfected, and waning back to them takes people out of the process
        S, V, I, R = ss.states(4)
        dis = ss.Disease(S >> I @ ss.perday(0.3)*I, S >> V @ ss.perday(0.02), I >> R @ ss.days(10), R >> S @ ss.days(30), init_n=dict(I=20), name='sirs')
        sim = ss.Sim(dis, total_pop=1e6, branching=True, solver=solver, unit='day', stop=60, verbose=0).run()
        assert np.isclose(sim.results.sirs.S[-1], (1e6 - 20)*np.exp(-0.02*60), rtol=1e-3), solver
        assert any('go back to the uninfected' in note for note in sim.notes), solver


def test_until():
    sc.heading('A run stopped by a condition: an expression of the model, or a function of the sim')
    sir = make_sir(n=10)
    sim = ss.Sim(sir, total_pop=1e6, branching=True, solver='gillespie', unit='day', stop=200, **kw)
    sim.run(until=sir.I.n + sir.R.n >= 500)
    I, R = sim.results.sir.I, sim.results.sir.R
    assert sim.complete and sim.stopped is not None and sim.branching.outcome == 'stopped'
    n = I[sim.stopped] + R[sim.stopped]
    assert n < 500 <= n + sim.results.sir.infection[sim.stopped] # The last time point with results: the condition held at the end of its step
    assert np.isnan(I[sim.stopped + 1:]).all()
    sim = ss.Sim(make_sir(n=10), n_agents=2000, unit='day', stop=200, **kw) # Not only for branching
    sim.run(until=lambda sim: sim.t.ti >= 20)
    assert sim.stopped == 20


def test_extinction():
    sc.heading('The chance a single case dies out is 1/R0, under the exact solver and as agents; offspring and generations as agents')
    for solver in ['gillespie', 'abm']:
        sir = make_sir()
        sim = ss.Sim(sir, total_pop=1e6, branching=True, solver=solver, unit='day', stop=100, **kw)
        b = ss.parallel(sim, n_runs=100, workers=1, run_args=dict(until=sir.I.n + sir.R.n >= 50)).branching # In series: each run takes a few milliseconds
        assert set(b.outcomes) == {'extinct', 'stopped'}, solver
        assert abs(b.p_extinct - 1/3) < 0.15, (solver, b.p_extinct) # About three standard errors
        assert b.R is None # Not fitted from runs stopped by until=, whose stop depends on their offspring

    # A gamma-distributed infectiousness gives negative binomial offspring, and each case has an infector and a generation
    k = 0.5
    S, I, R = ss.states(3)
    dis = ss.Disease(S >> I @ ss.perday(0.15)*I, I >> R @ ss.constant(ss.days(8)), init_n=dict(I=300), rel_trans=ss.gamma(shape=k, scale=1/k), name='sir')
    b = ss.Sim(dis, total_pop=1e7, branching=True, solver='abm', unit='day', stop=100, **kw).run().branching
    R0, kfit = ss.run.Branching.fit_offspring(b.offspring[b.complete])
    assert abs(R0 - 1.2) < 0.08 and abs(kfit - k) < 0.12, (R0, kfit)
    seeds = b.infector == -1
    assert seeds.sum() == 300 and np.all(b.generation[seeds] == 0)
    child = np.flatnonzero(~seeds)[0]
    assert b.generation[child] == b.generation[b.infector[child]] + 1
    assert b.length == b.generation.max() > 3


def test_ide():
    sc.heading('Exact dwell times (ss.solvers.ode(dwells="exact")): a dwell time as it is, and under branching, the renewal equation')
    def make(dur, solver, dt=1.0, branching=False, stop=150):
        S, E, I, R = ss.states(4)
        dis = ss.Disease(S >> E @ ss.perday(0.4)*I, E >> I @ dur, I >> R @ ss.perday(0.2), init_n=dict(I=10), name='seir')
        solver = ss.solvers.ode(dwells='exact') if solver == 'ide' else solver
        return ss.Sim(dis, total_pop=1e6, solver=solver, unit='day', dt=dt, stop=stop, branching=branching, **kw).run().results.seir
    gamma = ss.gamma(shape=4, scale=ss.days(2)) # Exact as four Erlang stages
    assert np.allclose(make(gamma, 'ode', 0.25).I.values, make(gamma, 'ide', 0.25).I.values, rtol=2e-3, atol=1)
    expon = ss.expon(mean=ss.days(4)) # As a conveyor, the same as one stage, including those who leave in the step they arrive in
    ode, ide = make(expon, 'ode').I.values, make(expon, 'ide').I.values
    assert np.abs(ode - ide).max() < 0.02*ode.max()
    delay = ss.constant(ss.days(6)) # Exact as a conveyor, as it already is
    assert np.array_equal(make(delay, 'ode').I.values, make(delay, 'ide').I.values)
    days = ss.choice(a=ss.days([3, 4, 5, 8]), p=[0.1, 0.4, 0.3, 0.2]) # Whole steps: a conveyor under every compartmental solver, not stages
    assert np.array_equal(make(days, 'ode').I.values, make(days, 'ide').I.values)
    assert np.array_equal(make(ss.choice(a=ss.days([6]), p=[1]), 'ode').I.values, make(delay, 'ode').I.values)
    res = make(gamma, 'ide', branching=True, stop=100) # Grows at the rate the generation interval gives (Euler-Lotka)
    y = res.E.values + res.I.values
    t = np.arange(len(y))
    r = np.polyfit(t[60:], np.log(y[60:]), 1)[0]
    r_true = spo.brentq(lambda r: 0.4*(1 + 2*r)**-4/(r + 0.2) - 1, 1e-6, 2)
    assert abs(r - r_true) < 1e-3, (r, r_true)


def test_clocks():
    sc.heading('Clocks (the time in a state) as agents only: in a split as people leave (HPV), and their errors under the compartmental solvers (a clock in a rate is in test_events.py)')
    logf2 = lambda x, k: 2/(1 + np.exp(-k*x)) - 1 # HPVsim's logistic of the dwell; NumPy only, so it works on expressions
    def hpv():
        P, C, R = ss.states('P C R')
        return ss.Disease(P, C, R, precin=ss.Transition(P, {C: logf2(P.time_in/ss.years(1), 0.5), R: None}, dur=ss.lognormal(ss.years(3), ss.years(2))), default_state='P', name='hpv')
    pars = dict(start=2000, stop=2030, dt=0.25, unit='year', **kw)
    sigma2 = np.log(1 + (2/3)**2) # The lognormal of mean 3 and sd 2, in whole steps as agents leave at the one nearest their dwell
    dwell = np.random.default_rng(0).lognormal(np.log(3) - sigma2/2, np.sqrt(sigma2), 1_000_000)
    q = logf2(np.maximum(np.round(dwell/0.25), 1)*0.25, 0.5).mean()
    assert abs(ss.Sim(hpv(), n_agents=50_000, **pars).run().results.hpv.C[-1]/50_000 - q) < 0.01
    with pytest.raises(ss.CapabilityError):
        ss.Sim(hpv(), solver='binomial', total_pop=50_000, **pars).init()
    assert isinstance(ss.Sim(hpv(), total_pop=50_000, **pars).init().solver, ss.solvers.abm)
    P, C, R = ss.states('P C R')
    with pytest.raises(ss.CapabilityError): # A branch with p= is decided on entry, so cannot read it
        ss.Disease(P, C, R, a=ss.Transition(P, C, dur=ss.years(3), p=P.time_in/10), b=ss.Transition(P, R, dur=ss.years(3), p=None), init_n=dict(P=10))

    # A modifier of a conveyor's rate cannot run as compartments
    def seir():
        S, E, I, R = ss.states('S E I R')
        return ss.Disease(S, E, I, R, infection=S >> E @ ss.perday(0.4)*I, latent=ss.Transition(E, I, dur=ss.constant(ss.days(14))), recovery=I >> R @ ss.days(7), init_prev=0.01, name='d')
    d = seir()
    with pytest.raises(ss.CapabilityError):
        ss.Sim(d, connectors=ss.modify(d.latent, multiply=2), solver='ode', **kw).run()


def test_solver_table():
    sc.heading('The five solvers with fixed delays: final size and endemic S and I against theory, and the peak against the ODE')
    stay, fixed = ss.days(5), ss.constant(ss.days(5))
    models = dict( # R0 = 2: the latent period, the time infectious, and the time immune
        exponential = (None, stay, None),
        fixed_stay  = (None, fixed, None),
        fixed_both  = (fixed, fixed, None),
        latent      = (fixed, stay, None),
        stay_latent = (stay, fixed, None),
        endemic     = (None, fixed, ss.days(20)),
    )
    theory = dict(final=0.7968, S=0.5, I=0.1) # 1 - z = exp(-2z); S* = 1/R0, and the rest are in I and R as 5 days are to 20
    tol = dict(final=0.05, S=0.05, I=0.1, peak=0.12)

    def run(solver, latent, infectious, waning):
        S, E, I, R = ss.states('S E I R')
        beta = ss.perday(0.4)*I
        flows = dict(infection=S >> I @ beta) if latent is None else dict(infection=S >> E @ beta, latency=ss.Transition(E, I, dur=latent))
        if waning is not None:
            flows['waning'] = R >> S @ waning
        n = 20_000 if solver == 'abm' else 100_000
        size = dict(n_agents=n, dt=0.25) if solver == 'abm' else dict(total_pop=n) # Agents count those infected in the step reported, so a short step
        dis = ss.Disease(recovery=ss.Transition(I, R, dur=infectious), **flows, init_prev=0.01, name='d')
        res = ss.Sim(dis, solver=solver, unit='day', stop=500 if waning else 300, rand_seed=1, **size, **kw).run().results.d
        return sc.objdict(final=res.R[-1]/n, S=res.S[-100:].mean()/n, I=res.I[-100:].mean()/n, peak=res.I.max()/n)

    for name,model in models.items():
        row = sc.objdict({solver:run(solver, *model) for solver in ['ode', 'binomial', 'tau', 'gillespie', 'abm']})
        for solver,got in row.items():
            for key,expected in ([('S', theory['S']), ('I', theory['I'])] if name == 'endemic' else [('final', theory['final'])]) + [('peak', row.ode.peak)]:
                allowed = tol[key]
                if solver == 'binomial' and key == 'peak':
                    allowed = 0.2 # A fixed delay doesn't add substeps, so a model with no other exit takes whole steps, which lowers its peak
                elif solver == 'binomial' and key == 'final' and name in ['fixed_stay', 'fixed_both', 'stay_latent']:
                    allowed = 0.015 # An odd number of substeps keeps the delay at its length, so the final size is close
                assert abs(got[key] - expected) <= allowed*expected, f'{name} under {solver}: {key} is {got[key]:n}, against {expected:n}'


if __name__ == '__main__':
    for name,test in list(globals().items()):
        if name.startswith('test_'):
            test()
