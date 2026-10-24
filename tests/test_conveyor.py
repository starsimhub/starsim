"""
Test the conveyor: fixed delays kept as levels of whole steps under the compartmental solvers; and clocks, which are agents only
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


@sc.timer()
def test_exit_split():
    sc.heading('A split taken by the dwell just completed (HPV: CIN from P.time), read as people leave: as agents only')
    k = 0.5
    def make():
        P, C, R = ss.states('P C R')
        p_cin = logf2(P.time/ss.years(1), k)
        return ss.Disease(P, C, R, precin=ss.Transition(P, [C, R], p=[p_cin, None], dur=ss.lognorm_ex(ss.years(3), ss.years(2))), first='P', name='hpv')
    kw2 = dict(start=2000, stop=2030, dt=0.25, unit='year', **kw)
    sigma2 = np.log(1 + (2/3)**2) # The lognormal of mean 3 and sd 2, in whole steps as agents leave at the one nearest their dwell
    dwell = np.random.default_rng(0).lognormal(np.log(3) - sigma2/2, np.sqrt(sigma2), 1_000_000)
    q = logf2(np.maximum(np.round(dwell/0.25), 1)*0.25, k).mean()
    abm = ss.Sim(make(), n_agents=50_000, **kw2).run().results.hpv
    assert abs(abm.C[-1]/50_000 - q) < 0.01
    for solver in ['ode', 'binomial']:
        with pytest.raises(ss.CapabilityError, match='P.time is a clock'):
            ss.Sim(make(), solver=solver, total_pop=50_000, **kw2).init()
    assert ss.Sim(make(), total_pop=50_000, **kw2).init().solver == 'abm'
    return abm


@sc.timer()
def test_syphilis_clock():
    sc.heading('A clock of a set of states as agents only: syphilis latent infectiousness by (EL | LL).time (run in test_clocks.py::test_syphilis_latent)')
    def make(**kwargs):
        S, E, EL, LL = ss.states('S E EL LL')
        syph = ss.Disease(S, E, EL, LL,
            infection = S >> E @ 0.8*(E + EL + LL),
            latency   = E >> EL @ ss.months(3),
            to_late   = EL >> LL @ ss.years(2),
            init_prev = 0.02, init=dict(EL=0.01, LL=0.01), name='syph')
        latent = (EL | LL).time
        decay = ss.modify(syph.rel_trans, multiply=0.5**(latent/ss.years(1)), where=EL | LL, name='latent_decay')
        return ss.Sim(syph, connectors=decay, start=2000, stop=2010, dt=ss.months(1), unit='year', **kw, **kwargs)
    with pytest.raises(ss.CapabilityError, match='is a clock'):
        make(solver='ode', total_pop=20_000).init()
    sim = make(total_pop=20_000).init()
    assert sim.solver == 'abm'
    return sim


@sc.timer()
def test_bv_clock():
    sc.heading('The time in a state in a rate, as agents only: BV recurrence by P.time')
    def cohort(): # A cohort that starts in P leaves at the hazard of its time in P: as agents, the hazard at the start of each step
        P, R = ss.states('P R')
        return ss.Disease(P, R, recur=P >> R @ ss.permonth(0.06)*np.exp(-P.time/ss.months(3)), first='P', name='bv')
    kw2 = dict(start=2000, stop=2003, dt=ss.months(1), unit='year', **kw)
    abm = ss.Sim(cohort(), n_agents=10_000, **kw2).run().results.bv
    h = 0.06*np.exp(-np.arange(len(abm.P))/3) # Per month, at the start of each month in P
    expected = 10_000*np.exp(-np.concatenate([[0], np.cumsum(h)[:-1]]))
    assert np.allclose(abm.P, expected, rtol=0.03)
    for solver in ['ode', 'gillespie']:
        with pytest.raises(ss.CapabilityError, match='P.time is a clock'):
            ss.Sim(cohort(), solver=solver, total_pop=10_000, **kw2).init()
    assert ss.Sim(cohort(), total_pop=10_000, **kw2).init().solver == 'abm'
    return abm


@sc.timer()
def test_conveyor_limits():
    sc.heading('A clock under a compartmental solver: a clear error, and agents under solver="auto"')
    def make():
        S, A, B = ss.states('S A B')
        return ss.Disease(S, A, B, sa=S >> A @ ss.perday(0.05), ab=ss.Transition(A, B, dur=ss.days(4)), ax=ss.Transition(A, S, rate=ss.perday(0.1)*A.progress), name='d')
    with pytest.raises(ss.CapabilityError, match='A.progress is a clock'): # .progress of a dwell time
        ss.Sim(make(), solver='ode', **kw).init()
    assert ss.Sim(make(), **kw).init().solver == 'abm'
    d = make_seir(init_prev=0.01)
    vx = ss.vaccinate(ss.Vaccine(efficacy=0), who=d.E & (d.E.time >= ss.days(2)), name='vx')
    with pytest.raises(ss.CapabilityError, match='E.time is a clock'): # A delivery reading a clock
        ss.Sim(d, interventions=vx, solver='ode', **kw).init()
    d = make_seir(init_prev=0.01)
    with pytest.raises(ss.CapabilityError, match='fixed delay and a modifier'): # A modifier of a conveyor's rate
        ss.Sim(d, connectors=ss.modify(d.latent, multiply=2), solver='ode', **kw).run()
    return


if __name__ == '__main__':
    sc.options(interactive=False)
    test_exit_split()
    test_syphilis_clock()
    test_bv_clock()
    test_conveyor_limits()
