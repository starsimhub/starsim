"""
Keeling and Rohani (2008), chapter 7: spatial structure (metapopulations, lattices, and networks)

Each model is written as the book describes it; `ref` integrates the book's equations with scipy. Program numbers and parameter values are from the review's notes (the book's program pages could not be fetched from this machine), so they follow the book's models rather than its exact listings.
"""
import numpy as np
import networkx as nx
import starsim as ss
from common import ode, run_all, series

beta, gamma = 1.0, 0.1 # Per day: R0 = 10 in every patch


#%% 7.1: patches coupled through the force of infection (rho_ij), nobody moves

rho = 0.01 # The share of each patch's force of infection that comes from the other patch
N1 = [5_000, 5_000]

def p71(**kw):
    """ 7.1: two patches coupled by rho: lambda_i = beta * sum_j rho_ij Y_j / N_j """
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ ss.perday(beta) * I, recovery=I >> R @ ss.perday(gamma), init=dict(I=ss.by(patch=[5, 0])), name='sir')
    sir.stratify(patch=['a', 'b'], split=[0.5, 0.5], mobility=ss.Mobility(matrix=[[0, rho], [rho, 0]], kind='force'))
    return ss.Sim(sir, unit='day', stop=100, total_pop=sum(N1), **kw)

def r71(sim):
    t = np.arange(101.)
    r = np.array([[1-rho, rho], [rho, 1-rho]])
    N = np.array(N1, float)
    def f(t, y):
        S, I = y[:2], y[2:]
        lam = beta * r @ (I/N)
        return np.concatenate([-lam*S, lam*S - gamma*I])
    y = ode(f, [N[0]-5, N[1], 5, 0], t)
    return dict(Ia=y[2], Ib=y[3], Sb=y[1])


#%% 7.2: explicit migration: people relocate at rate m_ij

m = np.array([[0, 0.002, 0.001], [0.002, 0, 0.001], [0.004, 0.004, 0]]) # Per day, from row to column
N2 = np.array([6_000, 3_000, 1_000.])

def p72(**kw):
    """ 7.2: three patches, people migrate between them; transmission within the current patch, frequency-dependent """
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ ss.perday(beta) * I, recovery=I >> R @ ss.perday(gamma), init=dict(I=ss.by(patch=[5, 0, 0])), name='sir')
    sir.stratify(patch=['a', 'b', 'c'], split=list(N2/N2.sum()), mobility=ss.Mobility(matrix=m, kind='migrate'))
    return ss.Sim(sir, unit='day', stop=100, total_pop=N2.sum(), **kw)

def r72(sim):
    t = np.arange(101.)
    out = m.sum(axis=1)
    def f(t, y):
        S, I, R = y[:3], y[3:6], y[6:]
        N = S + I + R
        lam = beta * I / N
        mig = lambda X: m.T @ X - out*X
        return np.concatenate([-lam*S + mig(S), lam*S - gamma*I + mig(I), gamma*I + mig(R)])
    y0 = np.concatenate([N2 - [5, 0, 0], [5, 0, 0], [0, 0, 0]])
    y = ode(f, y0, t)
    return dict(Ia=y[3], Ib=y[4], Ic=y[5], Nc=y[2]+y[5]+y[8])


#%% 7.3: commuters: residents of i spend a share of their time in j and return; the time-share limit, and the book's explicit X_ij form

away = 0.1 # Share of time residents spend in the other patch
N3 = np.array([8_000, 2_000.])
T3 = np.array([[1-away, away], [away, 1-away]])

def p73(**kw):
    """ 7.3: two patches coupled by commuting (residents keep their home patch) """
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ ss.perday(beta) * I, recovery=I >> R @ ss.perday(gamma), init=dict(I=ss.by(patch=[5, 0])), name='sir')
    sir.stratify(patch=['a', 'b'], split=list(N3/N3.sum()), mobility=ss.Mobility(matrix=[[0, away], [away, 0]], kind='commute'))
    return ss.Sim(sir, unit='day', stop=100, total_pop=N3.sum(), **kw)

def r73(sim):
    """ The time-share form: lambda_i = sum_k T_ik beta (sum_j T_jk I_j) / (sum_j T_jk N_j) """
    t = np.arange(101.)
    def f(t, y):
        S, I = y[:2], y[2:]
        lam = T3 @ (beta * (T3.T @ I) / (T3.T @ N3))
        return np.concatenate([-lam*S, lam*S - gamma*I])
    y = ode(f, [N3[0]-5, N3[1], 5, 0], t)
    return dict(Ia=y[2], Ib=y[3])

def r73_explicit(l=0.5):
    """ The book's explicit form: X_ij residents of i now in j, leaving home at rate l_ij and returning at rate r_ij, with r chosen to give the same share of time away """
    r = l*(1-away)/away
    t = np.arange(101.)
    def f(t, y):
        S = y[:4].reshape(2, 2); I = y[4:].reshape(2, 2)
        N = np.array([[N3[0]*(1-away), N3[0]*away], [N3[1]*away, N3[1]*(1-away)]]) # At the movement equilibrium
        lam = beta * I.sum(axis=0) / N.sum(axis=0) # Force in each place j
        def move(X):
            d = np.zeros((2, 2))
            for i in range(2):
                j = 1-i
                d[i, i] += r*X[i, j] - l*X[i, i]
                d[i, j] += l*X[i, i] - r*X[i, j]
            return d
        dS = -lam*S + move(S)
        dI = lam*S - gamma*I + move(I)
        return np.concatenate([dS.ravel(), dI.ravel()])
    S0 = np.array([[N3[0]*(1-away)-5*(1-away), N3[0]*away-5*away], [N3[1]*away, N3[1]*(1-away)]])
    I0 = np.array([[5*(1-away), 5*away], [0, 0]])
    y = ode(f, np.concatenate([S0.ravel(), I0.ravel()]), t)
    return dict(Ia=y[4]+y[5], Ib=y[6]+y[7])


#%% 7.4: a lattice of patches, each coupled to its four neighbors through the force of infection

L = 5
eps = 0.05 # The share of each patch's force that comes from its neighbors

def lattice():
    n = L*L
    M = np.zeros((n, n))
    for i in range(L):
        for j in range(L):
            k = i*L + j
            nbrs = [(i+di)*L + (j+dj) for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)] if 0 <= i+di < L and 0 <= j+dj < L]
            for q in nbrs:
                M[k, q] = eps/4 # Edge patches have fewer neighbors, and keep the rest of their own force
    return M

def p74(**kw):
    """ 7.4: a 5x5 lattice of equal patches, seeded in a corner """
    M = lattice()
    names = [f'p{k}' for k in range(L*L)]
    seed = [10] + [0]*(L*L-1)
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ ss.perday(beta) * I, recovery=I >> R @ ss.perday(gamma), init=dict(I=ss.by(patch=seed)), name='sir')
    sir.stratify(patch=names, mobility=ss.Mobility(matrix=M, kind='force'))
    return ss.Sim(sir, unit='day', stop=150, total_pop=L*L*1_000, **kw)

def r74(sim):
    M = lattice()
    n = L*L
    R_ = np.eye(n)*(1 - M.sum(axis=1)) + M.T # lambda_i = (1 - sum_k M_ik) F_i + sum_k M_ki F_k, as the spec says
    N = np.full(n, 1_000.)
    t = np.arange(151.)
    def f(t, y):
        S, I = y[:n], y[n:]
        lam = R_ @ (beta*I/N)
        return np.concatenate([-lam*S, lam*S - gamma*I])
    I0 = np.zeros(n); I0[0] = 10
    y = ode(f, np.concatenate([N - I0, I0]), t)
    return dict(I0=y[n], Icorner=y[n + n-1], Itotal=y[n:].sum(axis=0))


#%% 7.6: SIR on a random regular network, and the pair approximation

k_deg = 4
tau = 0.3 # Transmission rate per S-I pair per day
n_net = 10_000

def p76(**kw):
    """ 7.6: SIR on a random regular network with 4 contacts each, as agents """
    kw.setdefault('n_agents', n_net)
    kw.setdefault('solver', 'abm')
    g = nx.random_regular_graph(k_deg, kw['n_agents'], seed=1)
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(tau)), recovery=ss.progression('I -> R', rate=ss.perday(gamma)), init_prev=0.01, name='sir')
    kw.setdefault('dt', 0.25) # At a daily step the agents' epidemic peaks 7.5% lower and 4 days later than the pair approximation (repros/kr7_network_daily_step.py)
    return ss.Sim(sir, networks=ss.StaticNet(graph=g), unit='day', stop=120, **kw)

def r76(sim):
    """ The pair approximation with no clustering (phi = 0), in shares of the population """
    t = np.arange(481) * 0.25 # A quarter-day grid, as p76 runs
    n, I0 = k_deg, 0.01
    def f(t, y):
        S, I, SS, SI = y
        SSI = (n-1)/n * SS*SI/S
        ISI = (n-1)/n * SI*SI/S
        return [-tau*SI, tau*SI - gamma*I, -2*tau*SSI, tau*(SSI - ISI - SI) - gamma*SI]
    S0 = 1-I0
    y = ode(f, [S0, I0, n*S0*S0, n*S0*I0], t)
    N = sim.results.n_alive.values[0] if 'n_alive' in sim.results else n_net
    return dict(I=y[1]*N, S=y[0]*N)

def c76(sim, ref):
    """ Distributional: compare peak size, peak time, and final size, relative """
    I, S = series(sim, 'sir.n_I'), series(sim, 'sir.n_S')
    return dict(peak=abs(I.max()/ref['I'].max()-1), peak_time=abs(I.argmax()-ref['I'].argmax())/ref['I'].argmax(), final=abs((1e4-S[-1])/(1e4-ref['S'][-1])-1))

def r76_meanfield(N=n_net):
    t = np.arange(121.)
    y = ode(lambda t, y: [-tau*k_deg*y[0]*y[1], tau*k_deg*y[0]*y[1] - gamma*y[1]], [0.99, 0.01], t)
    return dict(I=y[1]*N, S=y[0]*N)


models = {
    'KR 7.1 patches, force':    dict(build=p71, ref=r71, map={'sir.n_I|patch=a':'Ia', 'sir.n_I|patch=b':'Ib', 'sir.n_S|patch=b':'Sb'}, shows='Coupled epidemics, delay in the second patch'),
    'KR 7.2 migration':         dict(build=p72, ref=r72, map={'sir.n_I|patch=a':'Ia', 'sir.n_I|patch=b':'Ib', 'sir.n_I|patch=c':'Ic'}, shows='Explicit movement: patch sizes change'),
    'KR 7.3 commuters':         dict(build=p73, ref=r73, map={'sir.n_I|patch=a':'Ia', 'sir.n_I|patch=b':'Ib'}, shows='Commuting couples patches without mixing them'),
    'KR 7.4 lattice 5x5':       dict(build=p74, ref=r74, map={'sir.n_I|patch=p0':'I0', 'sir.n_I|patch=p24':'Icorner', 'sir.n_I':'Itotal'}, shows='A travelling wave across a lattice'),
    'KR 7.6 network pair appr': dict(build=p76, ref=r76, map={'sir.n_I':'I', 'sir.n_S':'S'}, check=c76, tol=0.1, fine=False, shows='Network SIR against the pair approximation (distributional)'),
}

if __name__ == '__main__':
    import sys
    run_all(models, select=sys.argv[1:] or None, others=())
