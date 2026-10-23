"""
Keeling and Rohani (2008), chapter 8: controlling infectious diseases (vaccination, isolation, contact tracing)

Each model is written as the book describes it; `ref` integrates the book's equations with scipy. Program numbers and parameter values are from the review's notes (the book's program pages could not be fetched from this machine).
"""
import numpy as np
import networkx as nx
import starsim as ss
from common import ode, run_all, series

beta, gamma, mu = 520/365, 1/7, 1/(70*365) # Measles-like, per day: R0 = beta/(gamma + mu) ~ 10
years = 50
S_eq = (gamma + mu)/beta # The endemic equilibrium with no vaccination, as the starting point
I_eq = mu*(1 - S_eq)/(gamma + mu) * 1 # mu (1 - S*) / beta S* simplifies to this
R_eq = 1 - S_eq - I_eq


def sir_demog(births, extra=None, **kw):
    """ An SIR with births and deaths at rate mu, in shares of the population, and these births """
    S, I, R = ss.states(3)
    N = (S + I + R).n
    trans = dict(
        infection = S >> I @ ss.perday(beta) * I,
        recovery  = I >> R @ ss.perday(gamma),
        deaths    = S + I + R >> ss.null @ ss.perday(mu),
    )
    trans.update(births(S, I, R, N))
    if extra:
        trans.update(extra(S, I, R, N))
    sir = ss.Disease(**trans, init=dict(S=S_eq, I=I_eq, R=R_eq), name='sir')
    return sir


#%% 8.1: vaccination of a share p of newborns

p81 = 0.7

def p81_(**kw):
    """ 8.1: a share p of newborns are vaccinated (born into R) """
    births = lambda S, I, R, N: dict(births=ss.null >> S @ ss.perday(mu*(1-p81)) * N, vaccinated=ss.null >> R @ ss.perday(mu*p81) * N)
    return ss.Sim(sir_demog(births), **(dict(unit='day', stop=years*365, total_pop=1.0) | kw))

def r81(sim, p=p81, v=0.0):
    t = np.arange(years*365 + 1.)
    def f(t, y):
        S, I = y
        return [mu*(1-p) - beta*S*I - mu*S - v*S, beta*S*I - (gamma+mu)*I]
    y = ode(f, [S_eq, I_eq], t)
    return dict(S=y[0], I=y[1])


#%% 8.2: vaccination of susceptibles at a constant rate v

v82 = 0.1/365

def p82(**kw):
    """ 8.2: susceptibles are vaccinated at rate v per day """
    births = lambda S, I, R, N: dict(births=ss.null >> S @ ss.perday(mu) * N)
    extra  = lambda S, I, R, N: dict(vaccination=S >> R @ ss.perday(v82))
    return ss.Sim(sir_demog(births, extra), **(dict(unit='day', stop=years*365, total_pop=1.0) | kw))

def r82(sim):
    return r81(sim, p=0.0, v=v82)


#%% 8.3: pulse vaccination: every T years, a share p of susceptibles is vaccinated

T83, p83 = 3, 0.5

def p83_(**kw):
    """ 8.3: pulses of vaccination of susceptibles every 3 years, starting at year 3 """
    births = lambda S, I, R, N: dict(births=ss.null >> S @ ss.perday(mu) * N)
    sir = sir_demog(births)
    pulse = ss.Delivery(move='S -> R', coverage=p83, start=T83*365, interval=T83*365, name='pulse')
    return ss.Sim(sir, interventions=pulse, **(dict(unit='day', stop=years*365, total_pop=1.0) | kw))

def r83(sim):
    """ Integrate between pulses, and move p of S to R at each pulse; as in v4's results, the stock on a pulse day is the one before the pulse (it is the state at the start of that day) """
    y = np.array([S_eq, I_eq])
    S, I = [y[0]], [y[1]]
    f = lambda t, y: [mu - beta*y[0]*y[1] - mu*y[0], beta*y[0]*y[1] - (gamma+mu)*y[1]]
    pulses = [T83*365*k for k in range(1, years//T83 + 1)]
    bounds = sorted(set([0] + pulses + [years*365]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        sol = ode(f, y, np.arange(a, b + 1.))
        S.extend(sol[0][1:]); I.extend(sol[1][1:])
        y = sol[:, -1].copy()
        if b in pulses:
            y[0] *= 1 - p83
    return dict(S=np.array(S), I=np.array(I))


#%% 8.4: isolation of cases (SIQR): the isolated do not mix, so they are out of the denominator

d84 = 0.2 # The rate at which cases are isolated, per day

def p84(**kw):
    """ 8.4: closed SIQR, isolated cases removed from mixing: lambda = beta I / (S + I + R) """
    S, I, Q, R = ss.states(4)
    dis = ss.Disease(
        infection = S >> I @ ss.perday(1.0) * I.n / (S + I + R).n,
        recovery  = I >> R @ ss.perday(0.1),
        isolation = I >> Q @ ss.perday(d84),
        release   = Q >> R @ ss.perday(0.1),
        init      = dict(I=1e-4),
        name = 'sir',
    )
    return ss.Sim(dis, **(dict(unit='day', stop=200, total_pop=1.0) | kw))

def r84(sim):
    t = np.arange(201.)
    def f(t, y):
        S, I, Q, R = y
        lam = 1.0*I/(S + I + R)
        return [-lam*S, lam*S - (0.1 + d84)*I, d84*I - 0.1*Q, 0.1*I + 0.1*Q]
    y = ode(f, [1-1e-4, 1e-4, 0, 0], t)
    return dict(S=y[0], I=y[1], Q=y[2])


#%% 8.5: contact tracing on a network (agents only): cases are found, and their contacts are vaccinated

def p85(trace=True, **kw):
    """ 8.5: SIR on a random regular network; each new case is found with probability 0.5 and their contacts are vaccinated """
    kw.setdefault('n_agents', 10_000)
    kw.setdefault('solver', 'abm')
    g = nx.random_regular_graph(4, kw['n_agents'], seed=1)
    net = ss.StaticNet(graph=g, name='net')
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.3)), recovery=ss.progression('I -> R', rate=ss.perday(0.1)), init_prev=0.002, name='sir')
    ivs = []
    if trace:
        ivs = ss.vaccinate(ss.Vaccine(efficacy=1.0), on=sir.infection, trace='net', coverage=0.5, name='ring')
    return ss.Sim(sir, networks=net, interventions=ivs, unit='day', dt=0.25, stop=120, **kw)

def r85(sim):
    """ No ODE reference: the check is that tracing runs and cuts the final size against no tracing """
    base = p85(trace=False, verbose=0).run()
    return dict(base_final=1e4 - series(base, 'sir.n_S')[-1])

def c85(sim, ref):
    final = 1e4 - series(sim, 'sir.n_S')[-1]
    print(f'    tracing: final size {final:.0f} against {ref["base_final"]:.0f} with none')
    return dict(reduction=0.0 if final < 0.8*ref['base_final'] else 1.0)


models = {
    'KR 8.1 newborn vaccination':  dict(build=p81_, ref=r81, map={'sir.n_S':'S', 'sir.n_I':'I'}, shows='Vaccinating newborns below the threshold lowers incidence, lengthens the interepidemic period'),
    'KR 8.2 constant vaccination': dict(build=p82, ref=r82, map={'sir.n_S':'S', 'sir.n_I':'I'}, shows='Vaccinating susceptibles at a constant rate'),
    'KR 8.3 pulse vaccination':    dict(build=p83_, ref=r83, map={'sir.n_S':'S', 'sir.n_I':'I'}, shows='Pulses of vaccination every T years'),
    'KR 8.4 isolation SIQR':       dict(build=p84, ref=r84, map={'sir.n_S':'S', 'sir.n_I':'I', 'sir.n_Q':'Q'}, shows='Isolation lowers R0 to beta/(gamma + d)'),
    'KR 8.5 contact tracing':      dict(build=p85, ref=r85, map={'sir.n_S':'base_final'}, check=c85, tol=0.5, fine=False, shows='Ring vaccination of the contacts of found cases (agents only)'),
}

if __name__ == '__main__':
    import sys
    run_all(models, select=sys.argv[1:] or None, others=())
