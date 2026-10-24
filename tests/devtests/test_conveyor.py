"""
Test the conveyor: fixed delays kept as levels of whole steps under the compartmental solvers; and clocks, which are agents only
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss

kw = dict(verbose=0)


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
    assert np.allclose(r.E[:14], n0*np.exp(-mu*k), rtol=1e-6) and r.E[14:].max() < 1e-9
    assert np.isclose(r.latent[14], n0*np.exp(-mu*13.5), rtol=1e-5) and np.isclose(r.latent.sum(), r.latent[14]) # Half of the last step's deaths come first
    abm = ss.Sim(cohort(), n_agents=10_000, unit='day', stop=20, **kw).run().results.d
    assert abm.latent[:14].sum() == 0 and abm.E[14] == 0 and np.isclose(abm.latent[14], n0*np.exp(-mu*14), rtol=0.05)

    # An epidemic: as agents, and converging to the exact delay as the step shrinks
    kw2 = dict(unit='day', stop=200, **kw)
    ode = ss.Sim(make_seir(init_prev=0.001), solver='ode', total_pop=50_000, **kw2).run().results.d
    abm = np.mean([ss.Sim(make_seir(init_prev=0.001), n_agents=50_000, rand_seed=seed, **kw2).run().results.d.R[-1] for seed in range(3)])
    assert np.isclose(ode.R[-1], abm, rtol=0.05), f'ODE {ode.R[-1]:n}, agents {abm:n}'
    fine = ss.Sim(make_seir(init_prev=0.001), solver='ode', total_pop=1.0, dt=0.25, **kw2).run().results.d
    peak, final = dde(14)
    assert np.isclose(fine.R[-1], final, rtol=0.01) and np.isclose(fine.I.max(), peak, rtol=0.03), f'{fine.R[-1]:.4f} vs {final:.4f}; peak {fine.I.max():.4f} vs {peak:.4f}'
    for solver in ['binomial', 'gillespie']: # Shifted at loop times under every solver
        res = ss.Sim(make_seir(init_prev=0.001), solver=solver, total_pop=50_000, **kw2).run().results.d
        assert np.isclose(res.R[-1], ode.R[-1], rtol=0.1), f'{solver}: {res.R[-1]:n} against {ode.R[-1]:n}'

    with pytest.raises(ValueError):
        ss.Transition(ss.State(), ss.State(), dur=1, stages='steps')
    return ode


if __name__ == '__main__':
    sc.options(interactive=False)
    test_fixed_delay()
