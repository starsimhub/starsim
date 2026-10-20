"""
Test the conveyor: clocks and dwell times kept as levels of whole steps under the compartmental solvers
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss

kw = dict(verbose=0)


def logf2(x, k):
    """ HPVsim's logistic of the dwell; NumPy only, so it works on expressions """
    return 2/(1 + np.exp(-k*x)) - 1


def make_seir(dur=ss.constant(ss.days(14)), beta=ss.perday(0.4), **kwargs):
    S, E, I, R = ss.states('S E I R')
    return ss.Disease(S, E, I, R,
        infection = S >> E @ beta*I,
        latent    = ss.Transition(E, I, dur=dur),
        recovery  = I >> R @ ss.days(7),
        name='d', **kwargs)


def dde(lag, beta=0.4, gamma=1/7, seed=0.001, stop=200, dt=0.01):
    """ The same SEIR with an exact delay, by Euler with a short step: the seeds start in E """
    n, k = int(stop/dt), int(round(lag/dt))
    s, i, r = 1 - seed, 0.0, 0.0
    inc = np.zeros(n)
    peak = 0
    for t in range(n):
        inc[t] = beta*s*i
        out = inc[t-k] if t >= k else 0.0
        s, i, r = s - dt*inc[t], i + dt*(out - gamma*i) + (seed if t == k else 0), r + dt*gamma*i
        peak = max(peak, i)
    return peak, r


@sc.timer()
def test_fixed_delay():
    sc.heading('Fixed delays: ss.constant as a conveyor of whole steps, exact on the step grid (odin, questions-oct12 #7)')

    # A cohort with a competing death: exact, as agents
    mu = 0.01
    def cohort():
        S, E, I = ss.states('S E I')
        return ss.Disease(S, E, I, infection=S >> E @ 0*I, latent=E >> I @ ss.constant(ss.days(14)), die=E >> ss.null @ ss.perday(mu), init=dict(E=0.1), name='d')
    k = np.arange(14) # The seeds are dated a step before the start, as if infected then, so they leave at day 14
    n0 = 1000
    ode = ss.Sim(cohort(), solver='ode', total_pop=10_000, unit='day', stop=20, **kw).run()
    r = ode.results.d
    assert np.allclose(r.n_E[:14], n0*np.exp(-mu*k), rtol=1e-6) and r.n_E[14:].max() < 1e-9
    assert np.isclose(r.latent[14], n0*np.exp(-mu*13.5), rtol=1e-5) and np.isclose(r.latent.sum(), r.latent[14]) # Half of the last step's deaths come first
    abm = ss.Sim(cohort(), n_agents=10_000, unit='day', stop=20, **kw).run().results.d
    assert abm.latent[:14].sum() == 0 and abm.n_E[14] == 0 and np.isclose(abm.latent[14], n0*np.exp(-mu*14), rtol=0.05)

    # An epidemic: as agents, and converging to the exact delay as the step shrinks
    kw2 = dict(unit='day', stop=200, **kw)
    ode = ss.Sim(make_seir(init_prev=0.001), solver='ode', total_pop=50_000, **kw2).run().results.d
    abm = np.mean([ss.Sim(make_seir(init_prev=0.001), n_agents=50_000, rand_seed=seed, **kw2).run().results.d.n_R[-1] for seed in range(3)])
    assert np.isclose(ode.n_R[-1], abm, rtol=0.05), f'ODE {ode.n_R[-1]:n}, agents {abm:n}'
    fine = ss.Sim(make_seir(init_prev=0.001), solver='ode', total_pop=1.0, dt=0.25, **kw2).run().results.d
    peak, final = dde(14)
    assert np.isclose(fine.n_R[-1], final, rtol=0.01) and np.isclose(fine.n_I.max(), peak, rtol=0.03), f'{fine.n_R[-1]:.4f} vs {final:.4f}; peak {fine.n_I.max():.4f} vs {peak:.4f}'
    for solver in ['binomial', 'gillespie']: # Shifted at loop times under every solver
        res = ss.Sim(make_seir(init_prev=0.001), solver=solver, total_pop=50_000, **kw2).run().results.d
        assert np.isclose(res.n_R[-1], ode.n_R[-1], rtol=0.1), f'{solver}: {res.n_R[-1]:n} against {ode.n_R[-1]:n}'

    with pytest.raises(ValueError):
        ss.Transition(ss.State(), ss.State(), dur=1, stages='steps')
    return ode


@sc.timer()
def test_exit_split():
    sc.heading('A split taken by the dwell just completed (HPV: CIN from P.time), read as people leave')
    k = 0.5
    def make():
        P, C, R = ss.states('P C R')
        p_cin = logf2(P.time/ss.years(1), k)
        return ss.Disease(P, C, R, precin=ss.Transition(P, [C, R], p=[p_cin, None], dur=ss.lognorm_ex(ss.years(3), ss.years(2))), name='hpv')
    kw2 = dict(start=2000, stop=2030, dt=0.25, unit='year', **kw)
    sigma2 = np.log(1 + (2/3)**2) # The lognormal of mean 3 and sd 2, in whole steps as agents leave at the one nearest their dwell
    dwell = np.random.default_rng(0).lognormal(np.log(3) - sigma2/2, np.sqrt(sigma2), 1_000_000)
    q = logf2(np.maximum(np.round(dwell/0.25), 1)*0.25, k).mean()
    ode = ss.Sim(make(), solver='ode', total_pop=1.0, **kw2).run().results.hpv
    assert np.isclose(ode.n_C[-1], q, rtol=0.01), f'{ode.n_C[-1]} vs {q}'
    abm = ss.Sim(make(), n_agents=50_000, **kw2).run().results.hpv
    assert abs(abm.n_C[-1]/50_000 - q) < 0.01
    bino = ss.Sim(make(), solver='binomial', total_pop=50_000, **kw2).run().results.hpv
    assert abs(bino.n_C[-1]/50_000 - q) < 0.01
    return ode


@pytest.mark.xfail(reason='Phase B, task B1: the time in a set of states ((EL | LL).time) as compartments', strict=True)
@sc.timer()
def test_syphilis_clock():
    sc.heading('Clocks as compartments: syphilis latent infectiousness by (EL | LL).time')
    hl = 1.0
    def make(beta=0.8, bins=None, init_prev=0.02, init=dict(EL=0.01, LL=0.01), **kwargs):
        S, E, EL, LL = ss.states('S E EL LL')
        syph = ss.Disease(S, E, EL, LL,
            infection = S >> E @ beta*(E + EL + LL),
            latency   = E >> EL @ ss.months(3),
            to_late   = EL >> LL @ ss.years(2),
            init_prev = init_prev, init=init, name='syph')
        latent = ss.clock(EL | LL, bins=bins)
        decay = ss.modify(syph.rel_trans, multiply=0.5**(latent/ss.years(hl)), where=EL | LL, name='latent_decay')
        return ss.Sim(syph, connectors=decay, start=2000, stop=2010, dt=ss.months(1), unit='year', **kw, **kwargs)

    # Exact: with almost no transmission, new infections come from the seeds, whose infectiousness halves each year, to the last level
    sim = make(beta=1e-6, init_prev=0, init=dict(LL=0.02), solver='ode', total_pop=10_000).run()
    conv = sim.diseases.syph.engine.conveyors[0]
    assert sim.solver == 'ode' and conv.kind == 'clock' and conv.steps == 1 and conv.n == 80 # Until 0.5**t is within 1% of where it ends
    inf = sim.results.syph.infection
    ratio = inf[2:]/inf[1:-1]
    assert np.allclose(ratio[:78], 0.5**(1/12), rtol=1e-5) and np.allclose(ratio[80:], 1, rtol=1e-5) # Held at the last level

    # As agents, and with wider levels
    ode = make(solver='ode', total_pop=20_000).run().results.syph
    abm = np.mean([make(n_agents=20_000, rand_seed=seed).run().results.syph.infection.sum() for seed in range(3)])
    assert np.isclose(ode.infection.sum(), abm, rtol=0.05), f'ODE {ode.infection.sum():n}, agents {abm:n}'
    wide = make(bins=ss.months(3), solver='ode', total_pop=20_000).run()
    assert wide.diseases.syph.engine.conveyors[0].steps == 3 and any('read at the mean time' in note for note in wide.notes)
    assert np.isclose(wide.results.syph.infection.sum(), ode.infection.sum(), rtol=0.05)
    return sim


@sc.timer()
def test_bv_clock():
    sc.heading('Clocks as compartments: BV recurrence by P.time')
    def make():
        class BV(ss.Disease):
            CST1, CST4, P = ss.states('CST1 CST4 P')
            recur0, recur_tau = ss.par(ss.permonth(0.06)), ss.par(ss.months(3))
            onset  = CST1 >> CST4 @ ss.permonth(0.05)
            tx     = CST4 >> P @ ss.months(2) # Treated after two months, on average
            recur  = ss.progression('P -> CST4', rate=recur0*np.exp(-P.time/recur_tau))
            settle = ss.progression('P -> CST1', dur=ss.months(6))
        return BV(name='bv')

    # A cohort that starts in P leaves at the hazard of its time in P: as agents, the hazard at the start of each step. As an ODE, each slot of time in P is read for people who arrived through a step (the usual case, and the one that converges as the step shrinks), so a cohort placed at the start is near the agents' answer, not the continuous one (0.835 at the end)
    def cohort():
        P, R = ss.states('P R')
        return ss.Disease(P, R, recur=P >> R @ ss.permonth(0.06)*np.exp(-P.time/ss.months(3)), name='bv')
    kw2 = dict(start=2000, stop=2003, dt=ss.months(1), unit='year', **kw)
    r = ss.Sim(cohort(), solver='ode', total_pop=10_000, **kw2).run().results.bv
    h = 0.06*np.exp(-np.arange(len(r.n_P))/3) # Per month, at the start of each month in P
    expected = 10_000*np.exp(-np.concatenate([[0], np.cumsum(h)[:-1]]))
    assert np.allclose(r.n_P, expected, rtol=0.01)
    abm = ss.Sim(cohort(), n_agents=10_000, **kw2).run().results.bv
    assert np.allclose(abm.n_P, expected, rtol=0.03)

    # The full model: as agents, at weekly and monthly steps (people who arrive in P at the start of a step face its hazard, the largest, for that step)
    kw2 = dict(start=2000, stop=2005, unit='year', **kw)
    ode = ss.Sim(make(), solver='ode', total_pop=50_000, dt=ss.weeks(1), **kw2).run().results.bv
    monthly = ss.Sim(make(), solver='ode', total_pop=50_000, dt=ss.months(1), **kw2).run().results.bv
    assert np.isclose(ode.recur.sum(), monthly.recur.sum(), rtol=0.03)
    for dt,ref in [(ss.weeks(1), ode), (ss.months(1), monthly)]:
        abm = ss.Sim(make(), n_agents=50_000, dt=dt, **kw2).run().results.bv
        assert np.isclose(ref.recur.sum(), abm.recur.sum(), rtol=0.03), f'{dt}: ODE {ref.recur.sum():n}, agents {abm.recur.sum():n}'
    for solver in ['binomial', 'gillespie']:
        res = ss.Sim(make(), solver=solver, total_pop=20_000, dt=ss.months(1), **kw2).run().results.bv
        assert np.isclose(res.recur.sum()*2.5, monthly.recur.sum(), rtol=0.1), solver
    return ode


@sc.timer()
def test_conveyor_limits():
    sc.heading('What a compartmental clock does not do yet: a clear error, and agents under solver="auto"')
    def make():
        S, A, B = ss.states('S A B')
        return ss.Disease(S, A, B, sa=S >> A @ ss.perday(0.05), ab=ss.Transition(A, B, dur=ss.days(4)), ax=ss.Transition(A, S, rate=ss.perday(0.1)*A.progress), name='d')
    with pytest.raises(ss.CapabilityError, match='A.progress is a clock'): # .progress of a dwell time
        ss.Sim(make(), solver='ode', **kw).init()
    assert ss.Sim(make(), **kw).init().solver == 'abm'
    d = make_seir(init_prev=0.01)
    vx = ss.vaccinate(ss.Vaccine(efficacy=0), who=d.E & (d.E.time >= ss.days(2)), name='vx')
    with pytest.raises(ss.CapabilityError, match='E.time is a clock'): # A delivery reading a clock (only an exit of its own state reads one as compartments)
        ss.Sim(d, interventions=vx, solver='ode', **kw).init()
    d = make_seir(init_prev=0.01)
    with pytest.raises(ss.CapabilityError, match='fixed delay and a modifier'): # A modifier of a conveyor's rate
        ss.Sim(d, connectors=ss.modify(d.latent, multiply=2), solver='ode', **kw).run()
    return


if __name__ == '__main__':
    sc.options(interactive=False)
    test_fixed_delay()
    test_exit_split()
    test_syphilis_clock()
    test_bv_clock()
    test_conveyor_limits()
