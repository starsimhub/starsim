"""
Keeling and Rohani (2008), chapter 5: temporally forced models (programs 5.1 to 5.3, and the bifurcation diagram of 5.1)

Each model is written as the book describes it; `ref` integrates the book's equations with scipy. Parameter values follow the book's programs as the reviewer recalls them (the program pages could not be fetched from this machine); the comparison is with the same equations, so it does not depend on that.
"""
import numpy as np
import starsim as ss
from common import ode, run_all

w = 2*np.pi/365 # The book's year is 365 days


#%% 5.1: SIR with sinusoidal forcing of beta: beta(t) = beta0 (1 + beta1 sin(w t))

b0, b1, g, m = 17/13, 0.1, 1/13, 1/(50*365)

def p51(beta1=b1, yrs=20, **kw):
    """ 5.1: SIR with births and deaths, and beta forced sinusoidally (sin peaks a quarter of a year in) """
    S, I, R = ss.states(3)
    sir = ss.Disease(
        infection = S >> I @ ss.seasonal(mean=ss.perday(b0), amplitude=beta1, peak=0.25) * I,
        recovery  = I >> R @ ss.perday(g),
        births    = ss.null >> S @ ss.perday(m) * (S + I + R).n,
        deaths    = S + I + R >> ss.null @ ss.perday(m),
        init      = dict(S=1/17, I=1e-4, R=1-1/17-1e-4),
        name = 'sir',
    )
    return ss.Sim(sir, **{**dict(unit='day', stop=yrs*365, total_pop=1.0), **kw})

def f51(beta1):
    def f(t, y):
        beta = b0*(1 + beta1*np.sin(w*t))
        return [m - beta*y[0]*y[1] - m*y[0], beta*y[0]*y[1] - (g+m)*y[1]]
    return f

def r51(sim, beta1=b1, yrs=20):
    y = ode(f51(beta1), [1/17, 1e-4], np.arange(yrs*365+1.))
    return dict(S=y[0], I=y[1])


#%% 5.1, the multiannual regime: measles-like SEIR with strong forcing, where the attractor is biennial (or longer)

def p51b(beta1=0.35, yrs=10, **kw):
    """ 5.1 (bifurcation): SEIR with births, beta forced sinusoidally with a large amplitude """
    mu, sig, gam, beta0 = 0.02/365, 1/8, 1/5, 1250/365
    S, E, I, R = ss.states(4)
    seir = ss.Disease(
        infection = S >> E @ ss.seasonal(mean=ss.perday(beta0), amplitude=beta1, peak=0.25) * I,
        latency   = E >> I @ ss.perday(sig),
        recovery  = I >> R @ ss.perday(gam),
        births    = ss.null >> S @ ss.perday(mu) * (S + E + I + R).n,
        deaths    = S + E + I + R >> ss.null @ ss.perday(mu),
        init      = dict(S=0.06, E=1e-4, I=1e-4, R=1-0.06-2e-4),
        name = 'sir',
    )
    return ss.Sim(seir, **{**dict(unit='day', stop=yrs*365, total_pop=1.0), **kw})

def f51b(beta1):
    mu, sig, gam, beta0 = 0.02/365, 1/8, 1/5, 1250/365
    def f(t, y):
        S, E, I = y
        beta = beta0*(1 + beta1*np.sin(w*t))
        return [mu - beta*S*I - mu*S, beta*S*I - (mu+sig)*E, sig*E - (mu+gam)*I]
    return f

def r51b(sim, beta1=0.35, yrs=10):
    y = ode(f51b(beta1), [0.06, 1e-4, 1e-4], np.arange(yrs*365+1.))
    return dict(S=y[0], E=y[1], I=y[2])


#%% 5.2: SEIR with term-time forcing: beta is high in term and low in the holidays

holidays = np.zeros(365, bool)
for a, b in [(0, 6), (100, 115), (200, 251), (300, 307), (356, 364)]:
    holidays[a:b+1] = True
t_mu, t_sig, t_gam, t_b0, t_b1 = 0.02/365, 1/8, 1/5, 1250/365, 0.25

def term(day):
    return 1 - t_b1 if holidays[int(day) % 365] else 1 + t_b1

def p52(yrs=20, **kw):
    """ 5.2: SEIR with births, and beta switched between term and holiday values on fixed days of the year """
    S, E, I, R = ss.states(4)
    seir = ss.Disease(
        infection = S >> E @ (lambda sim: ss.perday(t_b0*term(sim.ti))) * I,
        latency   = E >> I @ ss.perday(t_sig),
        recovery  = I >> R @ ss.perday(t_gam),
        births    = ss.null >> S @ ss.perday(t_mu) * (S + E + I + R).n,
        deaths    = S + E + I + R >> ss.null @ ss.perday(t_mu),
        init      = dict(S=0.06, E=1e-4, I=1e-4, R=1-0.06-2e-4),
        name = 'sir',
    )
    return ss.Sim(seir, **{**dict(unit='day', stop=yrs*365, total_pop=1.0), **kw})

def p52b(yrs=20, **kw):
    """ 5.2, declared: the term calendar as a yearly, piecewise-constant time series """
    lo, hi = 1 - t_b1, 1 + t_b1
    calendar = ss.timeseries({0: lo, 7: hi, 100: lo, 116: hi, 200: lo, 252: hi, 300: lo, 308: hi, 356: lo}, mode='constant', period=ss.days(365))
    S, E, I, R = ss.states(4)
    seir = ss.Disease(
        infection = S >> E @ ss.perday(t_b0) * calendar * I,
        latency   = E >> I @ ss.perday(t_sig),
        recovery  = I >> R @ ss.perday(t_gam),
        births    = ss.null >> S @ ss.perday(t_mu) * (S + E + I + R).n,
        deaths    = S + E + I + R >> ss.null @ ss.perday(t_mu),
        init      = dict(S=0.06, E=1e-4, I=1e-4, R=1-0.06-2e-4),
        name = 'sir',
    )
    return ss.Sim(seir, **{**dict(unit='day', stop=yrs*365, total_pop=1.0), **kw})


def r52(sim, yrs=20, shift=0):
    """ Day by day, since beta is constant within a day; shift=1 uses the next day's value (to detect an off-by-one) """
    y = np.zeros((3, yrs*365+1)); y[:, 0] = [0.06, 1e-4, 1e-4]
    for d in range(yrs*365):
        beta = t_b0*term(d + shift)
        def f(t, y):
            S, E, I = y
            return [t_mu - beta*S*I - t_mu*S, beta*S*I - (t_mu+t_sig)*E, t_sig*E - (t_mu+t_gam)*I]
        y[:, d+1] = ode(f, y[:, d], np.array([d, d+1.]))[:, -1]
    return dict(S=y[0], E=y[1], I=y[2])


#%% 5.3: SIR with sinusoidally forced births

def p53(yrs=20, **kw):
    """ 5.3: SIR whose birth rate (not beta) varies sinusoidally over the year; deaths are constant """
    S, I, R = ss.states(3)
    sir = ss.Disease(
        infection = S >> I @ ss.perday(b0) * I,
        recovery  = I >> R @ ss.perday(g),
        births    = ss.null >> S @ ss.seasonal(mean=ss.perday(m), amplitude=0.5, peak=0.25) * (S + I + R).n,
        deaths    = S + I + R >> ss.null @ ss.perday(m),
        init      = dict(S=1/17, I=1e-4, R=1-1/17-1e-4),
        name = 'sir',
    )
    return ss.Sim(sir, **{**dict(unit='day', stop=yrs*365, total_pop=1.0), **kw})

def r53(sim, yrs=20):
    def f(t, y):
        S, I, R = y; N = S + I + R
        return [m*(1 + 0.5*np.sin(w*t))*N - b0*S*I/N - m*S, b0*S*I/N - (g+m)*I, g*I - m*R]
    y = ode(f, [1/17, 1e-4, 1-1/17-1e-4], np.arange(yrs*365+1.))
    return dict(S=y[0], I=y[1], R=y[2])


models = {
    'KR 5.1 SIR sinusoidal beta': dict(build=p51, ref=r51, map={'sir.n_S':'S', 'sir.n_I':'I'}, shows='Forced oscillations, resonance with the natural period'),
    'KR 5.1 SEIR biennial':       dict(build=p51b, ref=r51b, map={'sir.n_S':'S', 'sir.n_I':'I'}, shows='Strong sinusoidal forcing gives a chaotic, multiannual attractor (compared over 10 years: beyond about 20, round-off grows, as in any integrator)'),
    'KR 5.2 SEIR term-time':      dict(build=p52, ref=r52, map={'sir.n_S':'S', 'sir.n_I':'I'}, shows='Term-time forcing of measles'),
    'KR 5.2 SEIR term-time (ts)': dict(build=p52b, ref=r52, map={'sir.n_S':'S', 'sir.n_I':'I'}, shows='Term-time forcing, as a yearly time series'),
    'KR 5.3 SIR forced births':   dict(build=p53, ref=r53, map={'sir.n_S':'S', 'sir.n_I':'I'}, shows='Seasonal births alone drive oscillations'),
}

def others(yrs=5, N=1e5, n_agents=10_000):
    """ Each model under the stochastic solvers, in counts, against the ODE in counts: peak and final number infected, over a shorter run """
    import io, contextlib
    from common import capture, series
    for name, m in models.items():
        base = m['build'](yrs=yrs, total_pop=N, verbose=0).run()
        I0 = series(base, 'sir.n_I')
        print(f'{name}: ode peak {I0.max():.4g}, final {I0[-1]:.4g}')
        for solver, kw in [('binomial', dict(total_pop=N)), ('gillespie', dict(total_pop=1e4)), ('abm', dict(n_agents=n_agents, total_pop=N))]:
            T = __import__('time').time()
            s, _, e = capture(lambda: m['build'](yrs=yrs, solver=solver, verbose=0, **kw).run())
            if e:
                print(f'    {solver:10s} error: {e[:200]}')
            else:
                I = series(s, 'sir.n_I'); scale = N/kw['total_pop']
                print(f'    {solver:10s} peak {I.max()*scale:.4g}, final {I[-1]*scale:.4g} (scaled to N), extinct={I[-1]==0}, {__import__("time").time()-T:.1f} s')


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['others']:
        others()
    else:
        run_all(models, select=sys.argv[1:] or None, others=())
