"""
Keeling and Rohani (2008), chapter 3: host heterogeneities (programs 3.1 to 3.4)

Each model is written as the book describes it, once with `stratify()` and a contact matrix where v4 offers that, and the reference integrates the book's equations with scipy. Parameter values follow the book's programs as best recalled (the program pages could not be fetched from this machine).
"""
import numpy as np
import starsim as ss
from common import ode, run_all

# 3.1: SIS with two risk groups (high, low), as for gonorrhoea; the force on group i is sum_j beta_ij I_j, with I_j the share of the whole population
b31 = np.array([[10, 0.1], [0.1, 1.0]]) # Per day
g31, nH = 1.0, 0.2
I0 = [1e-5, 1e-3] # Shares of the whole population

def p31(**kw):
    """ 3.1 as a stratified SIS with a contact matrix """
    sis = ss.Disease(infection=ss.transmission('S -> I', beta=1.0), recovery=ss.progression('I -> S', rate=ss.perday(g31)), init=dict(I=ss.by(risk=[I0[0]/nH, I0[1]/(1-nH)])), name='sis')
    sis.stratify(risk=['high', 'low'], split=[nH, 1-nH])
    n = np.array([nH, 1-nH])
    return ss.Sim(sis, networks=ss.contacts(risk=b31*n), unit='day', stop=30, total_pop=1.0, **kw) # Contacts with group j are beta_ij n_j, since the book's I_j is a share of everyone

def p31x(**kw):
    """ 3.1 written with explicit states, as the book's equations """
    SH, IH, SL, IL = ss.states(4, people=True)
    sis = ss.Disease(
        inf_H = SH >> IH @ ss.perday(b31[0,0]) * IH.n + ss.perday(b31[0,1]) * IL.n,
        inf_L = SL >> IL @ ss.perday(b31[1,0]) * IH.n + ss.perday(b31[1,1]) * IL.n,
        rec_H = IH >> SH @ ss.perday(g31),
        rec_L = IL >> SL @ ss.perday(g31),
        init  = dict(SH=nH-I0[0], IH=I0[0], SL=1-nH-I0[1], IL=I0[1]),
        name = 'sis',
    )
    return ss.Sim(sis, unit='day', stop=30, total_pop=1.0, **kw)

def r31(sim):
    t = np.arange(31.)
    def f(t, y):
        IH, IL = y
        return [b31[0] @ y * (nH - IH) - g31*IH, b31[1] @ y * (1-nH - IL) - g31*IL]
    y = ode(f, I0, t)
    return dict(IH=y[0], IL=y[1], I=y[0]+y[1])


# 3.2: SIS with five risk groups, proportionate (multiplicative) mixing, beta_ij = b * a_i a_j
a32 = np.array([0, 3, 10, 60, 100.]) / 30 # Activity of each group
n32 = np.array([0.06, 0.31, 0.52, 0.08, 0.03])
b32 = 0.0016 * 30 * np.outer(a32, a32)
g32 = 0.2
I032 = n32 * 1e-5 * 0 + np.array([0, 0, 0, 0, 1e-5])

def p32(**kw):
    sis = ss.Disease(infection=ss.transmission('S -> I', beta=1.0), recovery=ss.progression('I -> S', rate=ss.perday(g32)), init=dict(I=ss.by(risk=list(I032/n32))), name='sis')
    sis.stratify(risk=['r1', 'r2', 'r3', 'r4', 'r5'], split=list(n32))
    return ss.Sim(sis, networks=ss.contacts(risk=b32*n32), unit='day', stop=365, total_pop=1.0, **kw)

def r32(sim):
    t = np.arange(366.)
    def f(t, y):
        return b32 @ y * (n32 - y) - g32*y
    y = ode(f, I032, t)
    return dict(I=y.sum(0))


# 3.3: SIR with two age classes, children and adults; children age into adults at rate l, adults die at rate mu, births nu = mu * nA into children
b33 = np.array([[100, 10], [10, 20.]]) / 365
g33, l33, mu33 = 10/365, 1/(10*365), 1/(60*365)
nC = mu33/(l33 + mu33)
nu33 = mu33*(1-nC)
y033 = [0.1*nC, 1e-4*nC, 0.1*(1-nC), 1e-4*(1-nC)] # S_C, I_C, S_A, I_A, as shares of everyone

def p33(**kw):
    """ 3.3 with explicit states (aging between bands of a stratified age cannot be given this rate, see the README) """
    SC, IC, RC, SA, IA, RA = ss.states(6)
    sir = ss.Disease(
        inf_C   = SC >> IC @ ss.perday(b33[0,0]) * IC.n + ss.perday(b33[0,1]) * IA.n,
        inf_A   = SA >> IA @ ss.perday(b33[1,0]) * IC.n + ss.perday(b33[1,1]) * IA.n,
        rec_C   = IC >> RC @ ss.perday(g33),
        rec_A   = IA >> RA @ ss.perday(g33),
        age_S   = SC >> SA @ ss.perday(l33),
        age_I   = IC >> IA @ ss.perday(l33),
        age_R   = RC >> RA @ ss.perday(l33),
        births  = ss.null >> SC @ ss.perday(nu33),
        deaths  = SA + IA + RA >> ss.null @ ss.perday(mu33),
        init    = dict(SC=y033[0], IC=y033[1], RC=nC-y033[0]-y033[1], SA=y033[2], IA=y033[3], RA=1-nC-y033[2]-y033[3]),
        name = 'sir',
    )
    return ss.Sim(sir, unit='day', stop=5*365, total_pop=1.0, **kw)

def p33s(**kw):
    """ 3.3 with stratify(age=...) and a contact matrix, as v4 suggests: aging and deaths by band """
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=1.0), recovery=ss.progression('I -> R', rate=ss.perday(g33)),
                     deaths=ss.progression('S | I | R -> ', rate=ss.by(age=[ss.perday(0), ss.perday(mu33)])), init=dict(S=ss.by(age=[0.1, 0.1]), I=ss.by(age=[1e-4, 1e-4]), R=ss.by(age=[1-0.1-1e-4, 1-0.1-1e-4])), name='sir')
    sir.stratify(age=[0, 10], split=[nC, 1-nC])
    return ss.Sim(sir, networks=ss.contacts(age=b33*365*np.array([nC, 1-nC])/365), demographics=[ss.birth(rate=ss.peryear(nu33*365))], unit='day', stop=5*365, total_pop=1.0, **kw)

def r33(sim):
    t = np.arange(5*365+1.)
    def f(t, y):
        SC, IC, SA, IA = y
        lC = b33[0,0]*IC + b33[0,1]*IA
        lA = b33[1,0]*IC + b33[1,1]*IA
        return [nu33 - lC*SC - l33*SC, lC*SC - g33*IC - l33*IC, l33*SC - lA*SA - mu33*SA, l33*IC + lA*SA - g33*IA - mu33*IA]
    y = ode(f, y033, t)
    return dict(SC=y[0], IC=y[1], SA=y[2], IA=y[3], I=y[1]+y[3])


models = {
    'KR 3.1 SIS risk groups':          dict(build=p31, ref=r31, map={'sis.n_I':'I'}, shows='Core group sustains infection; high-risk prevalence'),
    'KR 3.1 SIS risk groups (states)': dict(build=p31x, ref=r31, map={'sis.n_IH':'IH', 'sis.n_IL':'IL'}, shows='The same, written as states'),
    'KR 3.2 SIS five risk groups':     dict(build=p32, ref=r32, map={'sis.n_I':'I'}, shows='Proportionate mixing across activity classes'),
    'KR 3.3 SIR two ages (states)':    dict(build=p33, ref=r33, map={'sir.n_SC':'SC', 'sir.n_IC':'IC', 'sir.n_IA':'IA'}, shows='Age structure with aging, WAIFW matrix'),
    'KR 3.3 SIR two ages (stratify)':  dict(build=p33s, ref=r33, map={'sir.n_I':'I'}, shows='The same with stratify(age=) and ss.contacts'),
}

if __name__ == '__main__':
    import sys
    run_all(models, select=sys.argv[1:] or None, others=())
