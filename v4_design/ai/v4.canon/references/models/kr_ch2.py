"""
Keeling and Rohani (2008), chapter 2: the simple models (programs 2.1 to 2.7)

Each model is written as the book describes it; `ref` integrates the book's equations with scipy. Parameter values follow the book's programs as published (from the review's notes; the program pages could not be fetched from this machine).
"""
import numpy as np
import starsim as ss
from common import ode, run_all

day = 1
yr = 365


def p21(**kw):
    """ 2.1: SIR, closed population, frequency-dependent """
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ ss.perday(1.4247) * I, recovery=I >> R @ ss.perday(0.14286), init=dict(I=1e-6), name='sir')
    return ss.Sim(sir, unit='day', stop=70, total_pop=1.0, **kw)

def r21(sim):
    t = np.arange(71.)
    y = ode(lambda t, y: [-1.4247*y[0]*y[1], 1.4247*y[0]*y[1] - 0.14286*y[1]], [1-1e-6, 1e-6], t)
    return dict(S=y[0], I=y[1])


mu22 = 1/(70*365)
def p22(**kw):
    """ 2.2: SIR with births and deaths at equal rates """
    S, I, R = ss.states(3)
    sir = ss.Disease(
        infection = S >> I @ ss.perday(520/365) * I,
        recovery  = I >> R @ ss.perday(1/7),
        births    = ss.null >> S @ ss.perday(mu22) * (S + I + R).n,
        deaths    = S + I + R >> ss.null @ ss.perday(mu22),
        init      = dict(S=0.1, I=1e-4, R=1-0.1-1e-4),
        name = 'sir',
    )
    return ss.Sim(sir, unit='day', stop=60*365, total_pop=1.0, **kw)

def r22(sim):
    b, g, m = 520/365, 1/7, mu22
    t = np.arange(60*365+1.)
    y = ode(lambda t, y: [m - b*y[0]*y[1] - m*y[0], b*y[0]*y[1] - (g+m)*y[1]], [0.1, 1e-4], t)
    return dict(S=y[0], I=y[1])


def p23(**kw):
    """ 2.3: SIR with disease-induced mortality (probability rho of dying before recovery), frequency-dependent, constant births nu """
    b, g, m, rho, nu = 520/365, 1/7, 1/(70*365), 0.5, 1/(70*365)
    X, Y, Z = ss.states(3)
    sir = ss.Disease(
        infection = X >> Y @ ss.perday(b) * Y,                # beta X Y / N
        recovery  = Y >> Z @ ss.perday(g),
        dying     = Y >> ss.null @ ss.perday(rho/(1-rho)*(g+m)),
        births    = ss.null >> X @ ss.perday(nu),
        deaths    = X + Y + Z >> ss.null @ ss.perday(m),
        init      = dict(X=0.2, Y=1e-4, Z=1-0.2-1e-4),
        name = 'sir',
    )
    return ss.Sim(sir, unit='day', stop=100*365, total_pop=1.0, **kw)

def r23(sim):
    b, g, m, rho, nu = 520/365, 1/7, 1/(70*365), 0.5, 1/(70*365)
    t = np.arange(100*365+1.)
    def f(t, y):
        X, Y, Z = y; N = X+Y+Z
        return [nu - b*X*Y/N - m*X, b*X*Y/N - (g+m)/(1-rho)*Y, g*Y - m*Z]
    y = ode(f, [0.2, 1e-4, 1-0.2-1e-4], t)
    return dict(S=y[0], I=y[1], R=y[2])


def p24(**kw):
    """ 2.4: SIR with disease-induced mortality, density-dependent transmission (beta X Y), in numbers """
    g, m, rho, nu = 1/7, 1/(70*365), 0.5, 1/(70*365)*1e4 # In numbers of hosts, with 10,000 at equilibrium
    X, Y, Z = ss.states(3)
    sir = ss.Disease(
        infection = X >> Y @ ss.perday(1e-4) * Y.n,
        recovery  = Y >> Z @ ss.perday(g),
        dying     = Y >> ss.null @ ss.perday(rho/(1-rho)*(g+m)),
        births    = ss.null >> X @ ss.perday(nu),
        deaths    = X + Y + Z >> ss.null @ ss.perday(m),
        init      = dict(X=5e3, Y=1, Z=5e3-1),
        name = 'sir',
    )
    return ss.Sim(sir, unit='day', stop=2*365, total_pop=1e4, **kw)

def r24(sim):
    g, m, rho, nu, b = 1/7, 1/(70*365), 0.5, 1/(70*365)*1e4, 1e-4
    t = np.arange(2*365+1.)
    def f(t, y):
        X, Y, Z = y
        return [nu - b*X*Y - m*X, b*X*Y - (g+m)/(1-rho)*Y, g*Y - m*Z]
    y = ode(f, [5e3, 1, 5e3-1], t)
    return dict(S=y[0], I=y[1], R=y[2])


def p25(**kw):
    """ 2.5: SIS """
    S, I = ss.states(2)
    sis = ss.Disease(infection=S >> I @ ss.perday(1.4247) * I, recovery=I >> S @ ss.perday(0.14286), init=dict(I=1e-6), name='sir')
    return ss.Sim(sis, unit='day', stop=70, total_pop=1.0, **kw)

def r25(sim):
    t = np.arange(71.)
    y = ode(lambda t, y: [-1.4247*y[0]*y[1] + 0.14286*y[1], 1.4247*y[0]*y[1] - 0.14286*y[1]], [1-1e-6, 1e-6], t)
    return dict(S=y[0], I=y[1])


def p26(**kw):
    """ 2.6: SEIR with births and deaths """
    m, b, s, g = 1/(70*365), 520/365, 1/14, 1/7
    S, E, I, R = ss.states(4)
    seir = ss.Disease(
        infection = S >> E @ ss.perday(b) * I,
        latency   = E >> I @ ss.perday(s),
        recovery  = I >> R @ ss.perday(g),
        births    = ss.null >> S @ ss.perday(m) * (S + E + I + R).n,
        deaths    = S + E + I + R >> ss.null @ ss.perday(m),
        init      = dict(S=0.1, E=1e-4, I=1e-4, R=1-0.1-2e-4),
        name = 'sir',
    )
    return ss.Sim(seir, unit='day', stop=60*365, total_pop=1.0, **kw)

def r26(sim):
    m, b, s, g = 1/(70*365), 520/365, 1/14, 1/7
    t = np.arange(60*365+1.)
    def f(t, y):
        S, E, I = y
        return [m - b*S*I - m*S, b*S*I - (m+s)*E, s*E - (m+g)*I]
    y = ode(f, [0.1, 1e-4, 1e-4], t)
    return dict(S=y[0], E=y[1], I=y[2])


def p27(**kw):
    """ 2.7: SIR with carriers: a fraction q of the infected become lifelong carriers, who transmit at a reduced rate epsilon """
    b, eps, g, G, m, q = 0.2, 0.1, 0.01, 0.001, 1/(50*365), 0.4
    S, I, C, R = ss.states(4)
    dis = ss.Disease(
        infection = S >> I @ ss.perday(b) * (I + eps * C),
        recovery  = I >> R @ ss.perday((1-q)*g),
        carriage  = I >> C @ ss.perday(q*g),
        clearance = C >> R @ ss.perday(G),
        births    = ss.null >> S @ ss.perday(m) * (S + I + C + R).n,
        deaths    = S + I + C + R >> ss.null @ ss.perday(m),
        init      = dict(S=0.1, I=1e-4, C=1e-3, R=1-0.1-1e-4-1e-3),
        name = 'sir',
    )
    return ss.Sim(dis, unit='day', stop=60*365, total_pop=1.0, **kw)

def r27(sim):
    b, eps, g, G, m, q = 0.2, 0.1, 0.01, 0.001, 1/(50*365), 0.4
    t = np.arange(60*365+1.)
    def f(t, y):
        S, I, C = y
        lam = b*(I + eps*C)
        return [m - lam*S - m*S, lam*S - (g+m)*I, q*g*I - (G+m)*C]
    y = ode(f, [0.1, 1e-4, 1e-3], t)
    return dict(S=y[0], I=y[1], C=y[2])


models = {
    'KR 2.1 SIR':              dict(build=p21, ref=r21, map={'sir.n_S':'S', 'sir.n_I':'I'}, shows='Epidemic curve, final size'),
    'KR 2.2 SIR demography':   dict(build=p22, ref=r22, map={'sir.n_S':'S', 'sir.n_I':'I'}, shows='Damped oscillations to endemic equilibrium'),
    'KR 2.3 SIR mortality fd': dict(build=p23, ref=r23, map={'sir.n_X':'S', 'sir.n_Y':'I', 'sir.n_Z':'R'}, shows='Disease-induced mortality, frequency-dependent'),
    'KR 2.4 SIR mortality dd': dict(build=p24, ref=r24, map={'sir.n_X':'S', 'sir.n_Y':'I', 'sir.n_Z':'R'}, shows='Disease-induced mortality, density-dependent (first 2 years: the ODE then falls to 1e-50 infected, below what any integrator resolves)'),
    'KR 2.5 SIS':              dict(build=p25, ref=r25, map={'sir.n_S':'S', 'sir.n_I':'I'}, shows='SIS equilibrium'),
    'KR 2.6 SEIR demography':  dict(build=p26, ref=r26, map={'sir.n_S':'S', 'sir.n_E':'E', 'sir.n_I':'I'}, shows='SEIR oscillations'),
    'KR 2.7 SIR carriers':     dict(build=p27, ref=r27, map={'sir.n_S':'S', 'sir.n_I':'I', 'sir.n_C':'C'}, shows='Carriers sustain infection'),
}

if __name__ == '__main__':
    import sys
    run_all(models, select=sys.argv[1:] or None, others=())
