"""
Vynnycky and White (2010), An Introduction to Infectious Disease Modelling: the book's model files (chapters 2 to 8), written in v4

The book's files are Berkeley Madonna and spreadsheet models; each is written here as the book describes it, with a scipy reference of the same equations. Parameter values are the book's as best recalled.
"""
import numpy as np
import starsim as ss
from common import ode, run_all

N = 1e5

# 2.1: SEIR in a closed population (influenza-like: R0 = 2, latent and infectious periods of 2 days), seeded with one case
def p21(R=0.0, **kw):
    S, E, I, Rr = ss.states('S E I R')
    seir = ss.Disease(infection=S >> E @ ss.perday(2/2) * I, latency=E >> I @ ss.days(2), recovery=I >> Rr @ ss.days(2), init=dict(I=1, R=R), name='seir')
    return ss.Sim(seir, unit='day', stop=150, total_pop=N, **kw)

def ref_seir(R0=2, De=2, Di=2, I0=1, R00=0.0, T=150):
    b = R0/Di
    def f(t, y):
        S, E, I = y
        return [-b*S*I/N, b*S*I/N - E/De, E/De - I/Di]
    y = ode(f, [N*(1-R00)-I0, 0, I0], np.arange(T+1.))
    return dict(S=y[0], E=y[1], I=y[2])

def r21(sim):
    return ref_seir()

# 3.x: the same with 40% immune before the epidemic (herd immunity, R_n = R0 * s)
def p3(**kw):
    return p21(R=0.4, **kw)
def r3(sim):
    return ref_seir(R00=0.4)

# 4.1: measles-like SEIR with births and deaths (R0 = 13, latent 8 days, infectious 7 days, life expectancy 70 years): the interepidemic period
De, Di, R0m, L = 8, 7, 13, 70*365
bm = R0m/Di # Ignoring the small deaths during infection
y041 = [N/R0m, 50, 50] # Start near the endemic number susceptible
def p41(**kw):
    S, E, I, R = ss.states(4)
    seir = ss.Disease(
        infection = S >> E @ ss.perday(bm) * I,
        latency   = E >> I @ ss.days(De),
        recovery  = I >> R @ ss.days(Di),
        births    = ss.null >> S @ ss.perday(1/L) * (S + E + I + R).n,
        deaths    = S + E + I + R >> ss.null @ ss.perday(1/L),
        init      = dict(S=y041[0], E=y041[1], I=y041[2], R=N-sum(y041)),
        name = 'seir',
    )
    return ss.Sim(seir, unit='day', stop=20*365, total_pop=N, **kw)

def r41(sim):
    def f(t, y):
        S, E, I = y
        return [N/L - bm*S*I/N - S/L, bm*S*I/N - E/De - E/L, E/De - I/Di - I/L]
    y = ode(f, y041, np.arange(20*365+1.))
    return dict(S=y[0], E=y[1], I=y[2])

# 5.1: MSEIR: newborns have maternal immunity for 6 months on average (here all mothers immune), then are susceptible; 5.2 adds vaccination of a share p of infants as they lose it
def p51(p=0.0, **kw):
    M, S, E, I, R = ss.states(5)
    seir = ss.Disease(
        waning    = M >> S @ ss.perday((1-p)/180),
        vaccinate = M >> R @ ss.perday(p/180),
        infection = S >> E @ ss.perday(bm) * I,
        latency   = E >> I @ ss.days(De),
        recovery  = I >> R @ ss.days(Di),
        births    = ss.null >> M @ ss.perday(1/L) * (M + S + E + I + R).n,
        deaths    = M + S + E + I + R >> ss.null @ ss.perday(1/L),
        init      = dict(M=N*180/L, S=y041[0], E=y041[1], I=y041[2], R=N-sum(y041)-N*180/L),
        name = 'seir',
    )
    return ss.Sim(seir, unit='day', stop=20*365, total_pop=N, **kw)

def r51(sim, p=0.0):
    def f(t, y):
        M, S, E, I = y
        return [N/L - M/180 - M/L, (1-p)*M/180 - bm*S*I/N - S/L, bm*S*I/N - E/De - E/L, E/De - I/Di - I/L]
    y = ode(f, [N*180/L] + y041, np.arange(20*365+1.))
    return dict(M=y[0], S=y[1], E=y[2], I=y[3])

def p52(**kw):
    return p51(p=0.6, **kw)
def r52(sim):
    return r51(sim, p=0.6)

# 7.1: two age groups (children, adults) with a WAIFW matrix, closed epidemic
nY = 0.25
waifw = np.array([[4.0, 1.0], [1.0, 2.0]]) * 1e-5 * 0.5 # beta_ij per day: the rate at which one infectious person in j infects one susceptible in i
Ny = np.array([nY, 1-nY])*N
def p71(**kw):
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=1.0), recovery=ss.progression('I -> R', dur=ss.days(7)), init=dict(I=ss.by(age=[10/Ny[0], 0])), name='sir')
    sir.stratify(age=[0, 15], split=[nY, 1-nY])
    return ss.Sim(sir, networks=ss.contacts(age=waifw*Ny), unit='day', stop=200, total_pop=N, **kw) # contacts with j: beta_ij * N_j, since the frequency-dependent rate divides by N_j

def r71(sim):
    def f(t, y):
        S, I = y[:2], y[2:]
        lam = waifw @ I
        return list(-lam*S) + list(lam*S - I/7)
    y = ode(f, [Ny[0]-10, Ny[1], 10, 0], np.arange(201.))
    return dict(I=y[2]+y[3], Iy=y[2])

# 8.1: an STI with a core group: partner change rates c_H, c_L, mixing between assortative (eps = 1) and proportionate (eps = 0), SIS
cH, cL, nH, eps, bs, Ds = 20/365, 2/365, 0.05, 0.5, 0.7, 180
n8 = np.array([nH, 1-nH]); c8 = np.array([cH, cL])
prop = c8*n8/(c8*n8).sum()
rho = eps*np.eye(2) + (1-eps)*np.tile(prop, (2, 1)) # rho_ij: the share of i's partners who are in j
C8 = c8[:, None] * rho # Partners per day of a person in i with people in j
def p81(**kw):
    sis = ss.Disease(infection=ss.transmission('S -> I', beta=bs), recovery=ss.progression('I -> S', dur=ss.days(Ds)), init=dict(I=ss.by(risk=[0.01, 0.01])), name='sti')
    sis.stratify(risk=['core', 'other'], split=list(n8))
    return ss.Sim(sis, networks=ss.contacts(risk=C8), unit='day', stop=20*365, total_pop=N, **kw)

def r81(sim):
    def f(t, y):
        I = y; Nn = n8*N
        lam = bs * C8 @ (I/Nn)
        return list(lam*(Nn - I) - I/Ds)
    y = ode(f, list(0.01*n8*N), np.arange(20*365+1.))
    return dict(I=y[0]+y[1], IH=y[0])


models = {
    'VW 2.1 SEIR closed':          dict(build=p21, ref=r21, map={'seir.n_I':'I', 'seir.n_S':'S'}, shows='Epidemic of an influenza-like infection'),
    'VW 3.x SEIR herd immunity':   dict(build=p3, ref=r3, map={'seir.n_I':'I', 'seir.n_S':'S'}, shows='Effect of prior immunity, R_n = R0 s'),
    'VW 4.1 SEIR measles endemic': dict(build=p41, ref=r41, map={'seir.n_I':'I', 'seir.n_S':'S'}, shows='Interepidemic period of measles'),
    'VW 5.1 MSEIR maternal':       dict(build=p51, ref=r51, map={'seir.n_M':'M', 'seir.n_I':'I'}, shows='Maternal immunity'),
    'VW 5.2 MSEIR vaccination':    dict(build=p52, ref=r52, map={'seir.n_M':'M', 'seir.n_I':'I'}, shows='Infant vaccination lengthens the interepidemic period'),
    'VW 7.1 WAIFW two ages':       dict(build=p71, ref=r71, map={'sir.n_I':'I'}, shows='Who acquires infection from whom'),
    'VW 8.1 STI core group':       dict(build=p81, ref=r81, map={'sti.n_I':'I'}, shows='Core group and assortativity'),
}

if __name__ == '__main__':
    import sys
    run_all(models, select=sys.argv[1:] or None, others=())
