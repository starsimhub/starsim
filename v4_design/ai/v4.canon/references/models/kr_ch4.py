"""
Keeling and Rohani (2008), chapter 4: multi-pathogen and multi-host models (programs 4.1 to 4.5)

Each model is written as the book describes it; the reference integrates the book's equations with scipy. Parameter values follow the book's programs as best recalled.
"""
import itertools
import numpy as np
import starsim as ss
from common import ode, run_all

# 4.1: two strains with partial cross-immunity (sigma reduces susceptibility to the other strain after recovery), with births and deaths
b1, b2, g, mu, sig = 3/7, 3.5/7, 1/7, 1/(5*365), 0.5
y041 = dict(S=0.98, I1=0.01, I2=0.01)

def p41(**kw):
    S, I1, I2, R1, R2, J1, J2, R12 = ss.states(8)
    two = ss.Disease(
        inf1   = S >> I1 @ ss.perday(b1) * (I1 + J1),
        inf2   = S >> I2 @ ss.perday(b2) * (I2 + J2),
        rec1   = I1 >> R1 @ ss.perday(g),
        rec2   = I2 >> R2 @ ss.perday(g),
        inf21  = R1 >> J2 @ ss.perday(b2*(1-sig)) * (I2 + J2), # J2: had strain 1, now infected with strain 2
        inf12  = R2 >> J1 @ ss.perday(b1*(1-sig)) * (I1 + J1),
        rec12  = J1 + J2 >> R12 @ ss.perday(g),
        births = ss.null >> S @ ss.perday(mu) * (S + I1 + I2 + R1 + R2 + J1 + J2 + R12).n,
        deaths = S + I1 + I2 + R1 + R2 + J1 + J2 + R12 >> ss.null @ ss.perday(mu),
        init   = y041,
        name = 'two',
    )
    return ss.Sim(two, unit='day', stop=3*365, total_pop=1.0, **kw)

def f41(t, y):
    S, I1, I2, R1, R2, J1, J2, R12 = y
    l1, l2 = b1*(I1 + J1), b2*(I2 + J2)
    return [mu - (l1+l2)*S - mu*S, l1*S - (g+mu)*I1, l2*S - (g+mu)*I2, g*I1 - (1-sig)*l2*R1 - mu*R1, g*I2 - (1-sig)*l1*R2 - mu*R2,
            (1-sig)*l1*R2 - (g+mu)*J1, (1-sig)*l2*R1 - (g+mu)*J2, g*(J1+J2) - mu*R12]

def r41(sim):
    y = ode(f41, [0.98, 0.01, 0.01, 0, 0, 0, 0, 0], np.arange(3*365+1.))
    return dict(I1=y[1]+y[5], I2=y[2]+y[6], S=y[0])

def p41s(**kw):
    """ 4.1 the v4 way: a strain dimension (kind='strain'), with cross-immunity as an adjustment """
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(b1)), recovery=ss.progression('I -> R', rate=ss.perday(g)),
                     init=dict(I=ss.by(strain=[0.01, 0.01])), name='two')
    sir.stratify(strain=['s1', 's2'], immunity=[[1, sig], [sig, 1]], adjust={'infection': [1, b2/b1]})
    return ss.Sim(sir, demographics=[ss.birth(rate=ss.perday(mu)), ss.death(rate=ss.perday(mu))], unit='day', stop=3*365, total_pop=1e4, **kw)


# 4.x: three strains, the exact 2^k history model: S_h is susceptible with history h (a set of strains had), I_hk is infected with strain k having history h
K = 3
beta3 = np.array([3, 3.5, 4.0])/7
hist = [frozenset(c) for r in range(K+1) for c in itertools.combinations(range(K), r)]
hn = lambda h: 'h' + (''.join(str(k+1) for k in sorted(h)) or '0')

def p4k(**kw):
    Sx = {h: ss.State(f'S_{hn(h)}') for h in hist}
    Ix = {(h, k): ss.State(f'I_{hn(h)}_{k+1}') for h in hist for k in range(K) if k not in h}
    trans = {}
    for k in range(K):
        Ik = [st for (h, kk), st in Ix.items() if kk == k]
        src = Ik[0]
        for st in Ik[1:]:
            src = src + st
        for h in hist:
            if k not in h:
                trans[f'inf_{hn(h)}_{k+1}'] = Sx[h] >> Ix[(h, k)] @ ss.perday(beta3[k]*(1-sig)**len(h)) * src
                trans[f'rec_{hn(h)}_{k+1}'] = Ix[(h, k)] >> Sx[h | {k}] @ ss.perday(g)
    init = {f'S_h0': 1 - 0.003} | {f'I_h0_{k+1}': 0.001 for k in range(K)}
    dis = ss.Disease(**trans, init=init, name='multi')
    return ss.Sim(dis, unit='day', stop=365, total_pop=1.0, **kw)

def r4k(sim):
    keysS = list(hist)
    keysI = [(h, k) for h in hist for k in range(K) if k not in h]
    nS = len(keysS)
    def f(t, y):
        S = dict(zip(keysS, y[:nS])); I = dict(zip(keysI, y[nS:]))
        lam = [beta3[k]*sum(v for (h, kk), v in I.items() if kk == k) for k in range(K)]
        dS = {h: -sum(lam[k]*(1-sig)**len(h) for k in range(K) if k not in h)*S[h] for h in keysS}
        dI = {}
        for (h, k) in keysI:
            dI[(h, k)] = lam[k]*(1-sig)**len(h)*S[h] - g*I[(h, k)]
            dS[h | {k}] += g*I[(h, k)]
        return [dS[h] for h in keysS] + [dI[x] for x in keysI]
    y0 = [1-0.003 if not h else 0 for h in keysS] + [0.001 if not h else 0 for (h, k) in keysI]
    y = ode(f, y0, np.arange(366.))
    out = {}
    for k in range(K):
        out[f'I{k+1}'] = sum(y[nS+i] for i, (h, kk) in enumerate(keysI) if kk == k)
    out['S0'] = y[0]
    return out

def multi_series(k):
    def get(sim):
        r = sim.results.multi
        return sum(np.asarray(r[f'n_I_{hn(h)}_{k}'].values) for h in hist if (k-1) not in h)
    return get


# 4.3: two host species, each SIR in a closed population, with transmission within and between species (frequency-dependent on the infected species' size)
B43 = np.array([[0.5, 0.05], [0.1, 0.3]]) # Per day: row is the infected species
N43 = np.array([10_000., 5_000.])
g43 = np.array([1/5, 1/10])

def p43(**kw):
    S1, I1, R1, S2, I2, R2 = ss.states(6, people=True)
    hosts = ss.Disease(
        inf1 = S1 >> I1 @ ss.perday(B43[0,0]) * I1.n / N43[0] + ss.perday(B43[0,1]) * I2.n / N43[1],
        inf2 = S2 >> I2 @ ss.perday(B43[1,0]) * I1.n / N43[0] + ss.perday(B43[1,1]) * I2.n / N43[1],
        rec1 = I1 >> R1 @ ss.perday(g43[0]),
        rec2 = I2 >> R2 @ ss.perday(g43[1]),
        init = dict(S1=N43[0]-10, I1=10, S2=N43[1], I2=0, R1=0, R2=0),
        name = 'hosts',
    )
    return ss.Sim(hosts, unit='day', stop=200, total_pop=N43.sum(), **kw)

def p43s(**kw):
    """ 4.3 with a species dimension and a contact matrix: contacts C_ij = beta_ij, since beta per contact is 1 """
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=1.0), recovery=ss.progression('I -> R', rate=ss.by(species=[ss.perday(g43[0]), ss.perday(g43[1])])),
                     init=dict(I=ss.by(species=[10/N43[0], 0])), name='hosts')
    sir.stratify(species=['cattle', 'deer'], split=list(N43/N43.sum()))
    return ss.Sim(sir, networks=ss.contacts(species=B43), unit='day', stop=200, total_pop=N43.sum(), **kw)

def r43(sim):
    def f(t, y):
        S1, I1, S2, I2 = y
        l1 = B43[0,0]*I1/N43[0] + B43[0,1]*I2/N43[1]
        l2 = B43[1,0]*I1/N43[0] + B43[1,1]*I2/N43[1]
        return [-l1*S1, l1*S1 - g43[0]*I1, -l2*S2, l2*S2 - g43[1]*I2]
    y = ode(f, [N43[0]-10, 10, N43[1], 0], np.arange(201.))
    return dict(I1=y[1], I2=y[3], I=y[1]+y[3])


# 4.4: a vector-borne infection: humans SIR, mosquitoes SI with births and deaths (Ross-Macdonald with the book's notation)
NH, NV = 10_000., 30_000.
a, bh, bv, gh, muv = 0.3, 0.5, 0.5, 1/14, 1/10

def p44(**kw):
    SH, IH, RH = ss.states(3)
    SV, IV = ss.states(2, people=False)
    vb = ss.Disease(
        human_inf = SH >> IH @ ss.perday(a*bh) * IV.n / NH,
        recovery  = IH >> RH @ ss.perday(gh),
        vector_inf = SV >> IV @ ss.perday(a*bv) * IH.n / NH,
        emergence = ss.null >> SV @ ss.perday(muv*NV),
        mortality = SV + IV >> ss.null @ ss.perday(muv),
        init = dict(IH=10, SV=NV),
        name = 'vb',
    )
    return ss.Sim(vb, unit='day', stop=365, total_pop=NH, **kw)

def r44(sim):
    def f(t, y):
        SH, IH, SV, IV = y
        lh, lv = a*bh*IV/NH, a*bv*IH/NH
        return [-lh*SH, lh*SH - gh*IH, muv*NV - lv*SV - muv*SV, lv*SV - muv*IV]
    y = ode(f, [NH-10, 10, NV, 0], np.arange(366.))
    return dict(IH=y[1], IV=y[3])


models = {
    'KR 4.1 two strains (states)':   dict(build=p41, ref=r41, map={'two.n_S':'S'}, shows='Partial cross-immunity; strain interference'),
    'KR 4.1 two strains (stratify)': dict(build=p41s, ref=r41, map={'two.n_S':'S'}, tol=0.02, scale=1e4, shows='The same with kind="strain"'),
    'KR 4.x three strains 2^k':      dict(build=p4k, ref=r4k, map={'multi.n_S_h0':'S0'}, shows='The exact history-based multistrain model D13 lacks'),
    'KR 4.3 two hosts (states)':     dict(build=p43, ref=r43, map={'hosts.n_I1':'I1', 'hosts.n_I2':'I2'}, shows='Spillover between host species'),
    'KR 4.3 two hosts (stratify)':   dict(build=p43s, ref=r43, map={'hosts.n_I':'I'}, shows='The same with a species dimension'),
    'KR 4.4 vector-borne':           dict(build=p44, ref=r44, map={'vb.n_IH':'IH', 'vb.n_IV':'IV'}, shows='Host-vector transmission'),
}

if __name__ == '__main__':
    import sys
    run_all(models, select=sys.argv[1:] or None, others=())
