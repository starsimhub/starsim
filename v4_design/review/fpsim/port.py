"""
FPsim (Kenya, SimpleChoice) ported to the Starsim v4 declarative API. Stress test, not production.

v3 sources (fpsim-v4 worktree, branch v4-port = fpsim main):
    fpsim/fpmod.py     FPmod(ss.Pregnancy): fecundity, sexual activity/debut, conception, abortion, miscarriage,
                       stillbirth, LAM, postpartum, parity, infant/maternal mortality
    fpsim/methods.py   SimpleChoice: logistic p_use by age x ever-used (x pp1/pp6), switching matrices by age,
                       method-specific duration-of-use distributions with age factors
    fpsim/interventions.py  update_methods, change_initiation(_prob), method_switching, add_method

Two formulations (see NOTES):
    make_sim()                 faithful: reproduction and contraception as two declared modules, semi-Markov method
                               durations, decisions on entry; agents only in practice (the ODE exhausts memory at init)
    make_compartmental_sim()   the same biology rearranged for compartments: one module, contraception as a dimension,
                               exponential dwells; runs as agents and as an ODE
Women only (FPsim's men are inert). Inputs are loaded by FPsim v3 itself, which runs unchanged on Starsim v4.

Run: conda run -n starsim python port.py
"""
import math
import numpy as np
import sciris as sc
import starsim as ss
from scipy.special import expit

BIG = ss.permonth(30) # "Move on the step the condition holds": the REVIEW §3 idiom; Euler with substeps=1 in the ODE
E = [0, 10, 15, 18, 20, 25, 30, 35, 40, 45, 50] # Age groups; must contain the method-choice edges 18, 20, 25, 35
PARITY = ['0', '1', '2', '3', '4+']
METHODS = ['pill', 'iud', 'inj', 'cond', 'btl', 'wdraw', 'impl', 'othtrad', 'othmod'] # FPsim order, idx 1..9
MODERN = ['pill', 'iud', 'inj', 'cond', 'btl', 'impl', 'othmod']


# %% Data: Kenya inputs, loaded and processed by FPsim v3 itself (which runs unchanged on Starsim v4), so both use the same numbers
def load_data():
    import fpsim as fp
    with sc.capture():
        sim = fp.Sim(location='kenya', n_agents=500, start=2000, stop=2001, verbose=0, contraception_module=fp.SimpleChoice(location='kenya'))
        sim.init()
    P = sim.people.fp.pars
    cm = sim.connectors.contraception
    D = sc.objdict(P=P, cm=cm)
    D.methods = {m.name: m for m in cm.methods.values()}
    D.mort_f = sim.demographics.deaths.pars.age_mortality['f_spline'] # Annual probability of death, women, by single year
    with sc.capture():
        D.pyramid = fp.get_dataloader('kenya').load(contra_mod='simple', return_data=True).people.age_pyramid # Age, men, women
    return D


def binned(arr, edges=None, top=None):
    """ Mean of a single-year-of-age array over each age group (the last group is edges[-1]..top) """
    edges = E if edges is None else edges
    arr = np.asarray(arr, dtype=float)
    top = top or len(arr)
    hi = list(edges[1:]) + [top]
    return np.array([arr[lo:min(h, len(arr))].mean() if lo < len(arr) else arr[-1] for lo,h in zip(edges, hi)])


def method_bin(edges=None):
    """ FPsim's method-choice age group (<18, 18-20, 20-25, 25-35, >35) of each age group """
    edges = E if edges is None else edges
    keys = ['<18', '18-20', '20-25', '25-35', '>35']
    return [keys[np.digitize(lo, [18, 20, 25, 35])] for lo in edges]


# %% Fertility tables: everything about conception that depends on age and parity, as one table per conception route
def p_month(raw):
    """ FPsim make_p_conceive: an annual probability, capped at 1, converted to a monthly one """
    return 1 - (1 - np.minimum(raw, 1))**(1/12)


def make_fertility(D, edges=None):
    """
    Hazards of conception per month, by (age, parity), for each reproductive state, without contraception, plus the
    multiplier each method applies to each of them. FPsim multiplies (1-efficacy) into an annual probability that is
    then capped at 1 and converted to a monthly one, so a method's effect on the hazard depends on age and parity
    (pill users aged 30 conceive at 0.4% of the no-method rate, not the 5.5% that 1-efficacy says).
    """
    edges = E if edges is None else edges
    P = D.P
    na = len(edges)
    fec = binned(P.age_fecundity[:51], edges)*0.9*P.exposure_factor*binned(P.exposure_age[:51], edges) # 0.9: mean of v3's personal fecundity, uniform(0.7, 1.1)
    nullip = binned(P.fecundity_ratio_nullip[:51], edges)
    k = len(PARITY) - 1
    exp_par = np.array([P.exposure_parity[i] for i in range(k)] + [P.exposure_parity[k:k+4].mean()]) # The top level is k+
    raw = fec[:,None]*exp_par[None,:]*np.where(np.arange(len(PARITY)) == 0, nullip[:,None], 1) # Annual probability, (age, parity)

    sa = binned(P.sexual_activity[:51], edges)
    sa_pp = P.sexual_activity_pp['percent_active']
    lam = P.lactational_amenorrhea['rate']*P.LAM_efficacy
    sp = P.spacing_pref['preference']
    space = lambda m0, m1: sp[np.minimum(np.arange(m0, m1)//3, P.spacing_pref['n_bins']-1)].mean() # v3 spacing weight, months since birth
    # Reproductive state: (activity by age or a constant, LAM effect, spacing weight). The postpartum bands are months since delivery
    routes = sc.objdict(
        NP = (np.where(np.arange(len(PARITY)) == 0, 1, space(51, 52))[None,:]*sa[:,None], 0, 1), # Never given birth, or 51+ months since
        PA = (sa_pp[0],          lam[0],          1), # 0-1 month: LAM; pp1 contraception decision on leaving
        PB = (sa_pp[1:6].mean(), lam[1:6].mean(), 1), # 1-6 months: LAM; pp6 decision on leaving
        PC = (sa_pp[6:18].mean(), 0,              1), # 6-18 months: v3 'postpartum' (dur_postpartum=18) activity
        PD = (sa[:,None],         0,              space(18, 39)),
        PE = (sa[:,None],         0,              space(39, 51)),
    )
    eff = {m: D.methods[m].efficacy for m in METHODS}
    haz, mult = {}, {}
    for name,(act,lam_eff,spacing) in routes.items():
        a = np.broadcast_to(np.asarray(act, dtype=float)*spacing, raw.shape) if np.ndim(act) else np.full(raw.shape, act*spacing)
        h0 = -np.log1p(-np.minimum(a*p_month(raw*(1-lam_eff)), 1-1e-9)) # Monthly hazard: v3 draws activity, then conception
        haz[name] = h0
        for m in METHODS:
            hm = -np.log1p(-a*p_month(raw*(1-lam_eff)*(1-eff[m])))
            mult[(name, m)] = np.divide(hm, h0, out=np.ones_like(h0), where=h0>0)
    return sc.objdict(haz=haz, mult=mult, sa=sa)


# %% The reproductive life course: a declared, non-infectious disease
def make_fp(D, edges=None, ode=False):
    edges = E if edges is None else edges
    P = D.P
    F = make_fertility(D, edges)
    st = dict(stages=4) if ode else {} # Fixed dwell times: in a compartmental run, Erlang stages (a constant has no variance to match)
    NP, P1, P2, PA, PB, PC, PD, PE = ss.states('NP P1 P2 PA PB PC PD PE') # Not pregnant, trimester 1, 2-3, postpartum bands
    INF = ss.State('INF', people=True) # Primary infertility. Without people=True it is guessed to be a count (only ss.null leads to it), and the run crashes in step_counts (see NOTES)
    rate = lambda name: ss.by(12*F.haz[name], dims=['age', 'parity']) # Per year, bare: ss.permonth(ss.by(...)) and a 2-D ss.by of typed rates both raise (see NOTES)

    # Pregnancy outcomes and their timing
    mc = binned(P.miscarriage_rates[:51], edges)
    h_mc = -np.log1p(-np.minimum(mc, 0.999))/ss.weeks(13).months # v3 spreads the total first-trimester risk over the trimester
    sb = P.stillbirth_rate
    sb_age = np.array([sb['age_probs'][np.argmin(np.abs(sb['ages'] - lo))] for lo in edges]) # v3: nearest age
    h_sb = -np.log1p(-sb['probs'][0]*sb_age)/ss.weeks(27).months # 2000 value (v3: nearest year; see NOTES)
    md = P.maternal_mortality['probs'][0]
    dur_p2 = ss.choice(a=ss.weeks(np.arange(32, 43) - 13), p=P.dur_pregnancy.pars.p) # v3 pregnancy length minus the first trimester
    twins = float(P.twins_prob)

    # Delivery moves parity up by one (two for twins): one transition per parity level and outcome, since move= takes one arrow
    deliver = {}
    for i,lev in enumerate(PARITY):
        for k,p in [(1, 1-twins), (2, twins)]:
            to = PARITY[min(i + k, len(PARITY)-1)]
            p_lev = ss.by(parity=p*(np.arange(len(PARITY)) == i)) # p=p raises E205: the check sums branches across parity levels, though move= keeps each to one level
            deliver[f'deliver_{lev}_{k}'] = ss.Transition(P2, PA, p=p_lev, dur=dur_p2, move=dict(parity=f'{lev} -> {to}'), **st)

    imr = P.infant_mortality['probs'][np.searchsorted(P.infant_mortality['year'], 2000)]
    mean_p2 = ss.weeks((np.arange(32, 43) - 13) @ P.dur_pregnancy.pars.p).years
    births_per_p2 = 0.5*(1 + twins)*(1 - imr)/mean_p2 # Surviving daughters per year per woman in P2: an inflow cannot be driven by a flow (see NOTES)

    conceive = dict(
        conceive    = NP >> P1 @ rate('NP'),
        conceive_pa = PA >> P1 @ rate('PA'),
        conceive_pb = PB >> P1 @ rate('PB'),
        conceive_pc = PC >> P1 @ rate('PC'),
        conceive_pd = PD >> P1 @ rate('PD'),
        conceive_pe = PE >> P1 @ rate('PE'),
    )
    pab = float(P.abortion_prob)
    conceive = {k: v*(1 - pab) for k,v in conceive.items()} # Abortions leave the woman where she was: the remaining 80% become pregnancies
    conceptions = ss.outcome(list(conceive.keys())) # By name: t1 + t2 adds the rates, and pab*t of a transition made outside a class body loses its name (E702; see NOTES)
    fp = ss.Disease(
        NP, INF, P1, P2, PA, PB, PC, PD, PE,
        **conceive,
        conceptions = conceptions, # Pregnancies carried on
        abortions = ss.outcome(pab/(1 - pab)*conceptions), # Counted, not tracked: nobody is marked as having had one
        tri1        = ss.Transition(P1, P2, dur=ss.constant(ss.weeks(13)), **st),
        miscarriage = P1 >> NP @ ss.by(age=12*h_mc),
        stillbirth  = P2 >> NP @ ss.by(age=12*h_sb),
        maternal    = P2 >> ss.null @ ss.permonth(-np.log1p(-md)/mean_p2/12),
        **deliver,
        pp1  = ss.Transition(PA, PB, dur=ss.constant(ss.months(1)), **st),
        pp6  = ss.Transition(PB, PC, dur=ss.constant(ss.months(5)), **st),
        pp18 = ss.Transition(PC, PD, dur=ss.constant(ss.months(12)), **st),
        pp39 = ss.Transition(PD, PE, dur=ss.constant(ss.months(21)), **st),
        pp51 = ss.Transition(PE, NP, dur=ss.constant(ss.months(12)), **st),
        births = ss.null >> (0.95*NP + 0.05*INF) @ ss.peryear(births_per_p2)*P2.n, # v3 p_infertile = 0.05
        deaths = ss.death(rate=ss.by(age=-np.log1p(-binned(D.mort_f, edges, top=100)))),
        init = dict(INF=0.05),
        name = 'fp',
    )
    fp.stratify(parity=PARITY, split=[1] + [0]*(len(PARITY) - 1)) # v3 starts every woman nulliparous (see NOTES)
    fp.F = F
    fp.pab = pab
    return fp


# %% Contraception: method states, semi-Markov switching, postpartum decisions
def p_use_table(cup, edges=None):
    """ v3 SimpleChoice.get_prob_use, as a table by (age group, ever used) """
    edges = E if edges is None else edges
    ab = np.digitize(edges, [18, 20, 25, 35])
    out = np.zeros((len(edges), 2))
    for e in [0, 1]:
        rhs = cup.intercept + cup.age_factors[ab] + e*cup.fp_ever_user + e*np.where(ab > 1, cup.age_ever_user_factors[np.maximum(ab-1, 0)], 0)
        out[:,e] = expit(rhs)
    out[np.array(edges) < 15] = 0 # Below method_age
    return out


def switch_row(D, event, agegrp, frm=None, jitter=1e-4):
    """ v3 choose_method: the destination shares among methods, zeros jittered, weighted and renormalized """
    mcp = D.cm.pars.method_choice_pars[event][agegrp]
    row = np.array(mcp[frm] if frm is not None else mcp, dtype=float)
    row = np.where(row > 0, row, jitter)*D.cm.pars.method_weights
    return row/row.sum()


def dur_use(D, m):
    """ A method's duration of use (months), and its age multipliers by v3 age bin. Reproduces v3's effective values (see NOTES) """
    mt = D.methods[m]
    d = mt.dur_use
    name = type(d).__name__
    af = np.asarray(getattr(d, 'age_factors', np.zeros(5)), dtype=float)
    if name == 'lognorm_ex': # v3 bug: 'lnorm' never matches 'lognorm', so these keep RandomChoice's 1 +/- 1 year and ignore their age factors
        return ss.lognorm_ex(ss.years(1), ss.years(1)), np.ones(5)
    if name == 'weibull':
        return ss.weibull(c=d.base_c, scale=ss.months(np.exp(d.base_scale))), np.exp(af)
    if name == 'gamma':
        return ss.gamma(a=np.exp(d.base_a), scale=ss.months(1/np.exp(d.base_scale))), np.exp(-af)
    if name == 'Fisk':
        return ss.weibull(c=np.exp(d.base_c), scale=ss.months(np.exp(d.base_scale))), np.exp(af) # v4 cannot take a custom Dist's stats (see NOTES); Weibull stand-in
    if name == 'uniform':
        return ss.uniform(ss.months(1000), ss.months(1200)), np.ones(5)
    raise NotImplementedError(name)


def make_contra(D, fp, edges=None, ode=False):
    edges = E if edges is None else edges
    cm = D.cm
    mb = method_bin(edges)
    abin = np.digitize(edges, [18, 20, 25, 35])
    pu = {k: p_use_table(cm.pars.contra_use_pars[i], edges) for i,k in enumerate(['pp0', 'pp1', 'pp6'])}
    names = ['none'] + METHODS
    pre, none, dec0, paused, dec1, wait6, dec6, *meth = ss.states(' '.join(['pre', 'none', 'dec0', 'paused', 'dec1', 'wait6', 'dec6'] + METHODS))
    S = dict(zip(names, [none] + meth))
    pregnant = fp.P1 | fp.P2
    tr = {}

    # Regular decision on leaving a method (or non-use): stop w.p. 1 - p_use(age, ever), else choose j from the row for the current method
    for frm in ['none'] + [m for m in METHODS if m != 'btl']: # BTL is permanent (v3 re-chooses BTL after 1000-1200 months)
        dist, age_mult = dur_use(D, frm)
        rows = np.array([switch_row(D, 0, k, frm) for k in mb])
        p_stop = 1 - pu['pp0']
        p_stop[np.array(edges) < 15] = 1
        tr[f'{frm}_to_none'] = ss.Transition(S[frm], none, p=ss.by(p_stop, dims=['age', 'ever_used']), dur=dist)
        for j,to in enumerate(METHODS):
            p = pu['pp0']*rows[:, j][:,None]
            tr[f'{frm}_to_{to}'] = ss.Transition(S[frm], S[to], p=ss.by(p, dims=['age', 'ever_used']), dur=dist, move=dict(ever_used=True))

    # Events from the reproductive module, as conditions on its states, at a big rate (REVIEW §3: no on= for transitions)
    tr['debut'] = ss.Transition(pre, dec0, rate=ss.by(age=debut_hazard(D, edges))) # v3: first choice at the fated debut age
    tr['pause'] = ss.Transition([none, dec0, wait6, dec6] + meth, paused, rate=BIG, where=pregnant) # v3 make_pregnancies: method = 0
    tr['resume'] = ss.Transition(paused, dec0, rate=BIG, where=fp.NP | fp.INF) # Pregnancy lost: choose again next step (v3 handle_loss)
    # Postpartum decisions at 1 and 6 months (v3 update_contra pp1/pp6). A p= branch gated by where= never fires (repro_where_branch.py),
    # so each decision is a hazard into a junction state, which then splits at once: one extra step each (see NOTES)
    rows1 = np.array([switch_row(D, 1, k) for k in mb])
    rows6 = np.array([switch_row(D, 6, k, 'none') for k in mb])
    tr['go_pp1'] = ss.Transition(paused, dec1, rate=BIG, where=fp.PA) # Seen the step of delivery; decided the step after, as v3 (ti_delivery + 1)
    tr['go_pp6'] = ss.Transition(wait6, dec6, rate=BIG, where=fp.PC)
    rows0 = np.array([switch_row(D, 0, k, 'none') for k in mb])
    for key, src, pk, rows, dflt in [('choose', dec0, 'pp0', rows0, none), ('pp1', dec1, 'pp1', rows1, wait6), ('pp6', dec6, 'pp6', rows6, none)]:
        tr[f'{key}_none'] = ss.Transition(src, dflt, p=ss.by(1 - pu[pk], dims=['age', 'ever_used']), rate=BIG) # rate=BIG: a p= branch with no rate crashes (REVIEW §6)
        for j,to in enumerate(METHODS):
            tr[f'{key}_{to}'] = ss.Transition(src, S[to], p=ss.by(pu[pk]*rows[:, j][:,None], dims=['age', 'ever_used']), rate=BIG, move=dict(ever_used=True))
    tr['program'] = ss.Transition(none, S['impl'], rate=ss.peryear(0), move=dict(ever_used=True)) # Driven only by the implant program (an undriven edge, REVIEW §3)

    # Initial state: FPsim draws p_use (never-users) for women past their fated debut, and the method from the initial mix
    F = debut_cdf(D, edges)
    init_mix = np.array([np.array(cm.pars.init_dist[k]) for k in mb])*cm.pars.method_weights
    init_mix /= init_mix.sum(1, keepdims=True)
    on = F*pu['pp0'][:,0]
    init = dict(none=ss.by(age=F*(1 - pu['pp0'][:,0]))) # By age only: an init by (age, ever_used) is silently zero (see NOTES)
    for j,m in enumerate(METHODS):
        init[m] = ss.by(age=on*init_mix[:,j])

    contra = ss.Disease(pre, none, dec0, paused, dec1, wait6, dec6, *meth, **tr, init=init, name='contra')
    contra.stratify(ever_used=[False, True], split=[1, 0], # So users at the start are marked one step late, and their first decision is drawn as never-users'
                    ever_mark=ss.transfer(ever_used=True, rate=BIG, states=METHODS))
    # Age factors on duration of use: an adjustment multiplies a transition's rate, so divides its duration (v3: scale = exp(base + factor))
    adjust = {}
    for frm in ['none'] + [m for m in METHODS if m != 'btl']:
        _, age_mult = dur_use(D, frm)
        for to in ['none'] + METHODS:
            adjust[f'{frm}_to_{to}'] = list(1/age_mult[abin])
    contra.stratify(age=edges, adjust=adjust)
    return contra


def debut_cdf(D, edges=None):
    """ Share of women past their fated debut age, at the middle of each age group """
    edges = E if edges is None else edges
    db = D.P.debut_age
    cdf = np.cumsum(db['probs'])/np.sum(db['probs'])
    mids = [(lo + hi)/2 for lo,hi in zip(edges, list(edges[1:]) + [edges[-1] + 10])]
    return np.interp(mids, db['ages'] + 0.5, cdf, left=0, right=1)


def debut_hazard(D, edges=None):
    """ Hazard of debut (per year) in each age group, from the fated-debut distribution (exact for a cohort) """
    edges = E if edges is None else edges
    db = D.P.debut_age
    surv = lambda a: 1 - np.interp(a, db['ages'] + 1, np.cumsum(db['probs'])/np.sum(db['probs']), left=0, right=1)
    hi = list(edges[1:]) + [edges[-1] + 10]
    return np.array([np.log(max(surv(lo), 1e-9)/max(surv(h), 1e-9))/(h - lo) if surv(h) > 1e-9 else 5.0 for lo,h in zip(edges, hi)])


# %% Connectors: what contraception and debut do to conception
def make_connectors(fp, contra):
    mods = []
    for route in ['NP', 'PA', 'PB', 'PC', 'PD', 'PE']:
        tname = 'conceive' if route == 'NP' else f'conceive_{route.lower()}'
        for m in METHODS: # 54 modifiers: the method's effect depends on age, parity and route (see make_fertility)
            mods.append(ss.modify(getattr(fp, tname), multiply=ss.by(fp.F.mult[(route, m)], dims=['age', 'parity']), where=getattr(contra, m), name=f'eff_{m}_{route}'))
    mods.append(ss.modify(fp.conceive, multiply=0, where=contra.pre, name='not_debuted')) # v3: nulliparous women are inactive before their fated debut
    return mods


# %% Interventions (FP programs)
def make_interventions(fp, contra, which):
    out = []
    if 'mix' in which: # v3 update_methods(method_mix=...) / method_weights: twice the injectable share from 2010. v4 takes the extra from every other branch, non-use included
        names = [n for n in contra.graph.transitions if n.endswith('_inj')]
        out += [ss.modify(getattr(contra, n), multiply=2.0, start=2010, name=f'mix_{n}') for n in names]
    if 'uptake' in which: # v3 change_initiation_prob: 30% fewer decisions not to use, from 2010 (the methods share the rest in proportion)
        names = [n for n in contra.graph.transitions if n.endswith('_none')]
        out += [ss.modify(getattr(contra, n), multiply=0.7, start=2010, name=f'uptake_{n}') for n in names]
    if 'implants' in which: # A count-based program: 100 implant insertions per year among non-users aged 20-35
        ppl = ss.People(1)
        out.append(ss.treat(ss.Treatment(efficacy=1.0, transition='program', name='implant'), eligible=contra.none & (ppl.age >= 20) & (ppl.age < 35),
                            capacity=ss.peryear(100), start=2010, interval=ss.months(1), disease='contra', name='implant_program')) # disease=: E506 with two declared modules
    return out


# %% Compartmental formulation: one module, contraception as a dimension of it, every dwell exponential
LEVELS = ['pre', 'never', 'none', 'pause0', 'pause1', 'wait0', 'wait1'] + METHODS # pause/wait: by ever used (0/1), which p_use needs afterwards
NOMETHOD = ['pre', 'never', 'none', 'pause0', 'pause1', 'wait0', 'wait1']


def exit_rate(D, m, edges=None):
    """
    An exponential hazard of leaving a method (per year), by age group, matched to the 12-month continuation of the
    agent version's distribution. Matching the mean instead fails for heavy tails: v3's implant Weibull (c = 0.29)
    has a median of 18 months and a mean of 55 years, so with the mean nobody would stop (see NOTES)
    """
    import scipy.stats as sps
    edges = E if edges is None else edges
    dist, mult = dur_use(D, m)
    p = dist.pars
    mo = lambda x: x.months if isinstance(x, ss.dur) else float(x)
    name = type(dist).__name__
    t = 12/mult[np.digitize(edges, [18, 20, 25, 35])] # An age multiplier on the scale is a divisor on time
    if name == 'lognorm_ex':
        mean, std = mo(p['mean']), mo(p['std'])
        sig = np.sqrt(np.log(1 + (std/mean)**2))
        surv = sps.lognorm(s=sig, scale=mean*np.exp(-sig**2/2)).sf(t)
    elif name == 'weibull':
        surv = np.exp(-(t/mo(p['scale']))**float(p['c']))
    elif name == 'gamma':
        surv = sps.gamma(a=float(p['a']), scale=mo(p['scale'])).sf(t)
    else: # BTL
        surv = np.ones_like(t)
    return -np.log(np.maximum(surv, 1e-9)) # Per year, since the window is one year


def make_compartmental(D, edges=None):
    """
    The same biology written so that a compartmental solver can take it: no p= branches (each would add a
    sub-compartment per branch per state), no second module (crossing two modules multiplies their compartments),
    exponential dwells. Contraception is a dimension of the reproductive module, moved by ss.transfer and by move= on
    the reproductive transitions (conception sets 'pause', the 1- and 6-month postpartum transitions choose a method).
    """
    edges = E if edges is None else edges
    P, cm = D.P, D.cm
    F = make_fertility(D, edges)
    mb = method_bin(edges)
    pu = {k: p_use_table(cm.pars.contra_use_pars[i], edges) for i,k in enumerate(['pp0', 'pp1', 'pp6'])}
    NP, P1, P2, PA, PB, PC, PD, PE = ss.states('NP P1 P2 PA PB PC PD PE')
    INF = ss.State('INF', people=True)
    nonpreg = ['NP', 'INF', 'PA', 'PB', 'PC', 'PD', 'PE']
    pab = float(P.abortion_prob)
    yr = lambda per_month: 12*per_month # Rates are bare numbers per year inside ss.by
    tr = {}

    # Conception, one transition per (route, method level): the method's effect is in the rate, and move= starts the pause
    for route, src in [('NP', NP), ('PA', PA), ('PB', PB), ('PC', PC), ('PD', PD), ('PE', PE)]:
        for lev in LEVELS:
            if lev == 'pre' or (lev.startswith('pause') and route not in ['PA']): # Not debuted; paused women only exist pregnant or just delivered
                continue
            mult = F.mult[(route, lev)] if lev in METHODS else 1.0
            to = 'pause0' if lev in ['never', 'pause0', 'wait0'] else 'pause1'
            tr[f'c_{route}_{lev}'] = ss.Transition(src, P1, rate=ss.by(yr(F.haz[route]*mult)*(1 - pab), dims=['age', 'parity']), move=dict(method=f'{lev} -> {to}'))

    # Pregnancy and its outcomes; after a loss, back to (ever-)non-use
    mc = binned(P.miscarriage_rates[:51], edges)
    sb = P.stillbirth_rate
    sb_age = np.array([sb['age_probs'][np.argmin(np.abs(sb['ages'] - lo))] for lo in edges])
    mean_p2 = ss.weeks((np.arange(32, 43) - 13) @ P.dur_pregnancy.pars.p)
    tr['tri1'] = P1 >> P2 @ ss.weeks(13)
    for k,(pz, back) in enumerate([('pause0', 'never'), ('pause1', 'none')]):
        tr[f'miscarriage_{k}'] = ss.Transition(P1, NP, rate=ss.by(age=yr(-np.log1p(-np.minimum(mc, .999))/ss.weeks(13).months)), move=dict(method=f'{pz} -> {back}'))
        tr[f'stillbirth_{k}'] = ss.Transition(P2, NP, rate=ss.by(age=yr(-np.log1p(-sb['probs'][0]*sb_age)/ss.weeks(27).months)), move=dict(method=f'{pz} -> {back}'))
    tr['maternal'] = P2 >> ss.null @ ss.permonth(-np.log1p(-P.maternal_mortality['probs'][0])/mean_p2.months)
    twins = float(P.twins_prob)
    for i,lev in enumerate(PARITY):
        for k,p in [(1, 1-twins), (2, twins)]:
            tr[f'deliver_{lev}_{k}'] = ss.Transition(P2, PA, rate=p/mean_p2, move=dict(parity=f'{lev} -> {PARITY[min(i+k, len(PARITY)-1)]}'))

    # Postpartum: the 1-month and 6-month decisions are the transitions out of PA and PB, split by destination as competing hazards
    rows1 = np.array([switch_row(D, 1, k) for k in mb])
    rows6 = np.array([switch_row(D, 6, k, 'none') for k in mb])
    for e in [0, 1]:
        tr[f'pp1_wait{e}'] = ss.Transition(PA, PB, rate=ss.by(age=yr(1 - pu['pp1'][:,e])), move=dict(method=f'pause{e} -> wait{e}'))
        tr[f'pp6_none{e}'] = ss.Transition(PB, PC, rate=ss.by(age=yr((1 - pu['pp6'][:,e])/5)), move=dict(method=f'wait{e} -> {"none" if e else "never"}'))
        for j,m in enumerate(METHODS):
            tr[f'pp1_{m}{e}'] = ss.Transition(PA, PB, rate=ss.by(age=yr(pu['pp1'][:,e]*rows1[:,j])), move=dict(method=f'pause{e} -> {m}'))
            tr[f'pp6_{m}{e}'] = ss.Transition(PB, PC, rate=ss.by(age=yr(pu['pp6'][:,e]*rows6[:,j]/5)), move=dict(method=f'wait{e} -> {m}'))
    for m in METHODS: # Women who chose a method at 1 month: a move= from a level to itself is the only way to say "people at this level" of the module's own dimension
        tr[f'pp6_{m}'] = ss.Transition(PB, PC, dur=ss.months(5), move=dict(method=f'{m} -> {m}'))
    tr['pp18'] = PC >> PD @ ss.months(12)
    tr['pp39'] = PD >> PE @ ss.months(21)
    tr['pp51'] = PE >> NP @ ss.months(12)
    imr = P.infant_mortality['probs'][np.searchsorted(P.infant_mortality['year'], 2000)]
    tr['births'] = ss.null >> (0.95*NP + 0.05*INF) @ ss.peryear(0.5*(1 + twins)*(1 - imr)/mean_p2.years)*P2.n
    tr['deaths'] = ss.death(rate=ss.by(age=-np.log1p(-binned(D.mort_f, edges, top=100))))

    # Switching: v3's decision at the end of a dwell, as competing hazards p*h, with h matched to 12-month continuation
    trans = {}
    h_debut = debut_hazard(D, edges)
    rows0 = np.array([switch_row(D, 0, k, 'none') for k in mb])
    trans['debut_never'] = ss.transfer(method='pre -> never', rate=ss.by(age=h_debut*(1 - pu['pp0'][:,0])), states=nonpreg)
    for j,m in enumerate(METHODS):
        trans[f'debut_{m}'] = ss.transfer(method=f'pre -> {m}', rate=ss.by(age=h_debut*pu['pp0'][:,0]*rows0[:,j]), states=nonpreg)
    for frm in ['never', 'none'] + [m for m in METHODS if m != 'btl']:
        key = 'none' if frm == 'never' else frm
        e = 0 if frm == 'never' else 1
        h = exit_rate(D, key, edges)
        rows = np.array([switch_row(D, 0, k, key) for k in mb])
        if frm not in ['never', 'none']:
            trans[f'{frm}_to_none'] = ss.transfer(method=f'{frm} -> none', rate=ss.by(age=(1 - pu['pp0'][:,e])*h), states=nonpreg)
        for j,m in enumerate(METHODS):
            if m != frm:
                trans[f'{frm}_to_{m}'] = ss.transfer(method=f'{frm} -> {m}', rate=ss.by(age=pu['pp0'][:,e]*rows[:,j]*h), states=nonpreg)

    # Initial method mix, as a split by (age, method level)
    Fd = debut_cdf(D, edges)
    init_mix = np.array([np.array(cm.pars.init_dist[k]) for k in mb])*cm.pars.method_weights
    init_mix /= init_mix.sum(1, keepdims=True)
    split = np.zeros((len(edges), len(LEVELS)))
    split[:, 0] = 1 - Fd
    split[:, 1] = Fd*(1 - pu['pp0'][:,0])
    split[:, len(NOMETHOD):] = (Fd*pu['pp0'][:,0])[:,None]*init_mix
    split[:, 0] += 1 - split.sum(1) # Rounding
    split *= age_split(D, edges)[:,None] # A split over (age, method) is read as joint, not as shares within each age: without this, every age group gets the same number of people (see NOTES)

    fpc = ss.Disease(NP, INF, P1, P2, PA, PB, PC, PD, PE, **tr, init=dict(INF=0.05), name='fp')
    fpc.stratify(parity=PARITY, split=[1] + [0]*(len(PARITY) - 1))
    fpc.stratify(method=LEVELS, split=ss.by(split, dims=['age', 'method']), **trans)
    fpc.pab = pab
    return fpc


def age_split(D, edges=None):
    """ Women by age group, from the Kenya 2000 pyramid (5-year bins to 80+) """
    edges = E if edges is None else edges
    single = np.repeat(D.pyramid[:,2]/5, 5)
    split = np.array([single[lo:hi].sum() for lo,hi in zip(edges, list(edges[1:]) + [len(single)])])
    return split/split.sum()


def make_compartmental_sim(D, solver='ode', n_agents=5000, total_pop=None, edges=None, parity=None, implants=False, **kwargs):
    global E, PARITY
    if edges is not None: E = edges
    if parity is not None: PARITY = parity
    fpc = make_compartmental(D)
    intvs = []
    if implants: # The same 100 insertions a year, as a capacity-limited transfer along the method dimension. The arrow form
        ppl = ss.People(1) # (method='none -> impl') raises KeyError '"n" is not a level' (repro_transfer_arrow.py), so: a target level plus eligible=
        intvs.append(ss.transfer(method='impl', eligible=(fpc.method == 'none') & (ppl.age >= 20) & (ppl.age < 35), capacity=ss.peryear(100),
                                 start=2010, interval=ss.months(1), disease='fp', name='implant_program'))
    kw = dict(start=2000, stop=2020, dt=ss.months(1), unit='year', verbose=0, solver=solver, rand_seed=1, stratify=dict(age=E, split=age_split(D)), interventions=intvs)
    if solver == 'ode':
        kw.update(total_pop=total_pop or n_agents, integrator='euler', substeps=1) # Euler, one substep: 'BIG' rates empty a compartment in one step, as agents do
    else:
        kw.update(n_agents=n_agents)
    kw.update(kwargs)
    return ss.Sim(fpc, **kw)


# %% Sim
def make_sim(D, solver='abm', n_agents=5000, total_pop=None, which=(), edges=None, parity=None, **kwargs):
    global E, PARITY
    if edges is not None: E = edges
    if parity is not None: PARITY = parity
    ode = solver == 'ode'
    fp = make_fp(D, ode=ode)
    contra = make_contra(D, fp, ode=ode)
    kw = dict(start=2000, stop=2020, dt=ss.months(1), unit='year', verbose=0, solver=solver, rand_seed=1, stratify=dict(age=E, split=age_split(D)))
    if ode:
        kw.update(total_pop=total_pop or n_agents, integrator='euler', substeps=1)
    else:
        kw.update(n_agents=n_agents)
    kw.update(kwargs)
    sim = ss.Sim([fp, contra], connectors=make_connectors(fp, contra), interventions=make_interventions(fp, contra, which), **kw)
    return sim


# %% Results
FP_STATES = ['NP', 'INF', 'P1', 'P2', 'PA', 'PB', 'PC', 'PD', 'PE']


def summarize(sim, label=None):
    """ mCPR and TFR as v3 defines them, method mix, parity, and event rates, for either formulation """
    r = sim.results
    f = r.fp
    yv = np.asarray(sim.t.yearvec)
    rep = [i for i,lo in enumerate(E) if 15 <= lo < 50]
    A = lambda res: np.asarray(res.by('age').values) # Time x age group
    one = 'contra' not in r
    def method(m): # Women at a method level, by age: a state of 'contra', or a level of the 'method' dimension of every reproductive state
        return sum(A(f[f'n_{s}'].select(method=m)) for s in FP_STATES) if one else A(r.contra[f'n_{m}'])
    w = sum(A(f[f'n_{s}']) for s in FP_STATES)
    mcpr = sum(method(m) for m in MODERN)[:, rep].sum(1)/np.maximum(w[:, rep].sum(1), 1e-9)
    dkeys = [k for k in f.keys() if k.startswith('deliver_')]
    deliv = sum(A(f[k]) for k in dkeys)
    widths = np.diff(E + [55])
    out = sc.objdict()
    for y in [2005, 2010, 2015, 2019]:
        i0, i1 = np.searchsorted(yv, y), np.searchsorted(yv, y + 1)
        asfr = deliv[i0:i1].sum(0)/np.maximum(w[i0:i1].mean(0), 1e-9) # Per woman per year
        out[f'tfr{y}'] = (asfr[rep]*widths[rep]).sum()
        out[f'mcpr{y}'] = mcpr[i0:i1].mean()
    out.mix = {m: method(m)[-1, rep].sum()/w[-1, rep].sum() for m in METHODS}
    if label is None:
        return out
    wy = w[:, rep].sum()/12 # Woman-years, 15-49
    pregs = sum(np.asarray(f[k]).sum() for k in f.keys() if k.startswith('conceive') or k.startswith('c_'))/(1 - sim.diseases.fp.pab)
    loss = lambda key: sum(np.asarray(f[k]).sum() for k in f.keys() if k.startswith(key))
    print(f'\n{label} (solver={sim.solver})')
    print('  mCPR 2005/2010/2015/2019: ' + ' / '.join(f'{out[f"mcpr{y}"]:.3f}' for y in [2005, 2010, 2015, 2019]))
    print('  TFR  2005/2010/2015/2019: ' + ' / '.join(f'{out[f"tfr{y}"]:.2f}' for y in [2005, 2010, 2015, 2019]))
    ab = f'abortions {1e3*np.asarray(f.abortions).sum()/wy:.1f}' if 'abortions' in f else 'abortions not tracked'
    print(f'  Per 1000 woman-years 15-49: pregnancies {1e3*pregs/wy:.0f}, births {1e3*deliv.sum()/wy:.0f}, miscarriages {1e3*loss("miscarriage")/wy:.1f}, stillbirths {1e3*loss("stillbirth")/wy:.1f}, {ab}')
    print('  Method mix at end (share of women 15-49): ' + ', '.join(f'{m} {v:.3f}' for m,v in out.mix.items()))
    par = sum(np.asarray(f[f'n_{s}'].by('parity').values)[-1] for s in FP_STATES)
    print('  Parity at end (everyone): ' + ', '.join(f'{p}: {v/par.sum():.2f}' for p,v in zip(PARITY, par)))
    return out


def v3_reference(n_agents=5000):
    """ The same configuration in FPsim v3, run on Starsim v4 """
    import fpsim as fp
    with sc.capture():
        sim = fp.Sim(location='kenya', n_agents=n_agents, start=2000, stop=2020, verbose=0, rand_seed=1, contraception_module=fp.SimpleChoice(location='kenya'))
        sim.run()
    r = sim.results
    yv = sim.t.yearvec
    yr = lambda y: slice(np.searchsorted(yv, y), np.searchsorted(yv, y+1))
    print('\nFPsim v3 on Starsim v4 (same inputs; both sexes)')
    print('  mCPR 2005/2010/2015/2019: ' + ' / '.join(f'{r.contraception.mcpr[yr(y)].mean():.3f}' for y in [2005, 2010, 2015, 2019]))
    print('  TFR  2005/2010/2015/2019: ' + ' / '.join(f'{r.fp.tfr[yr(y)].sum():.2f}' for y in [2005, 2010, 2015, 2019]))
    wy = r.fp.n_fecund.sum()/12 # Woman-years, 15-50
    print(f'  Per 1000 woman-years 15-49: pregnancies {1e3*r.fp.pregnancies.sum()/wy:.0f}, births {1e3*r.fp.births.sum()/wy:.0f}, '
          f'miscarriages {1e3*r.fp.miscarriages.sum()/wy:.1f}, stillbirths {1e3*r.fp.stillbirths.sum()/wy:.1f}, abortions {1e3*r.fp.abortions.sum()/wy:.1f}')
    mix = sim.people.fp.method_mix[:, -1]
    print('  Method mix at end (share of women 15-49): ' + ', '.join(f'{m} {v:.3f}' for m,v in zip(sim.connectors.contraception.methods.keys(), mix) if m != 'none'))
    return sim


if __name__ == '__main__':
    T = sc.timer()
    D = load_data()
    v3_reference()
    T.toc('v3')

    abm = make_sim(D, 'abm').run()
    T.toc('v4 faithful, agents')
    summarize(abm, 'v4 faithful: reproduction + contraception as two declared modules')
    base = summarize(abm)
    print('\n  Programs from 2010 (2019 values; baseline mCPR {:.3f}, TFR {:.2f}, injectables {:.3f}, implants {:.3f})'.format(base.mcpr2019, base.tfr2019, base.mix['inj'], base.mix['impl']))
    for which in ['mix', 'uptake', 'implants']:
        s = make_sim(D, 'abm', which=(which,)).run()
        o = summarize(s)
        extra = f', delivered {np.asarray(s.results.implant_program.delivered)[-60:].sum()/5:.0f}/year' if which == 'implants' else ''
        print(f'    {which:8s}: mCPR {o.mcpr2019:.3f}, TFR {o.tfr2019:.2f}, injectables {o.mix["inj"]:.3f}, implants {o.mix["impl"]:.3f}{extra}')
    T.toc('programs')

    coarse = dict(edges=[0, 15, 18, 20, 25, 35, 50], parity=['0', '1', '2+']) # The ODE takes ~1 s per simulated year at this size, ~30 s and 4.5 GB at full size (see NOTES)
    cabm = make_compartmental_sim(D, 'abm', **coarse).run()
    T.toc('v4 compartmental form, agents')
    summarize(cabm, 'v4 compartmental formulation (one module, method as a dimension), as agents')
    ode = make_compartmental_sim(D, 'ode', total_pop=5000, **coarse).run()
    T.toc('v4 compartmental form, ODE')
    summarize(ode, 'v4 compartmental formulation, as an ODE')
    prog = make_compartmental_sim(D, 'ode', total_pop=5000, implants=True, stop=2013, **coarse).run()
    print(f'\n  ODE implant program (capacity 100/year): delivered {np.asarray(prog.results.implant_program.delivered)[-36:].sum()/3:.1f} per year')
