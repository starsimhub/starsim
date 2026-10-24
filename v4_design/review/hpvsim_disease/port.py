"""
HPVsim v3 natural history, ported to the Starsim v4 API.

Structure (mirrors HPVsim v3: one disease module per genotype):
    hpv = make_hpv(g)    one ss.Disease per genotype, fully declared (states, transitions, outcomes)
    CrossImmunity        ss.Connector: per-agent, per-genotype nab/cell immunity, cross-genotype matrices, rel_sev
    make_hiv()           declared HIV with CD4 bands as states; effects on HPV via ss.modify
    HPVTotal             ss.Analyzer: any-genotype CIN prevalence by age, ASR cancer incidence

Natural history per genotype (females):
    S --infection--> P --dur_precin--> Pe --p(dur)--> C --dur_cin--> Ce --p(dur)--> X --dur_cancer--> dead
                                       |                             |
                                       +--> S (clear) | L (latent)   +--> S | L
    L --reactivation--> P;   males: P --dur_inf_male--> S
    Pe, Ce: zero-dwell junctions, needed so the CIN/cancer branch can depend on the sampled duration.
    Z: "cancer from another genotype", a sink.

Run: conda run -n starsim python port.py
"""
import itertools
import numpy as np
import sciris as sc
import starsim as ss

import warnings
warnings.filterwarnings('ignore', message='divide by zero encountered in divide', module='starsim.solvers') # where= on a dwell time divides by 0: see NOTES


# %% Severity functions, verbatim from hpvsim/utils.py (pure numpy)

def logf2(x, k, x_infl=0, ttc=50, y_max=1):
    t1 = (1 + np.exp(k*(x_infl - ttc)))
    t2 = (1 + np.exp(k*x_infl))
    u = y_max*t1*(1 - t2)/(t1 - t2)
    l = y_max*t1/(t1 - t2)
    return np.minimum(1, l + (u - l)/(1 + np.exp(k*(x_infl - x))))

def intlogf2(upper, k, x_infl=0, ttc=50, y_max=1):
    def indef(x):
        t1 = 1 + np.exp(k*(x_infl - ttc))
        t2 = 1 + np.exp(k*x_infl)
        return y_max/(t1 - t2)*(1 - t1*t2*(np.log(np.exp(k*(x_infl - x)) + 1)/k + x))
    upper = np.asarray(upper, dtype=float)
    out = indef(np.minimum(ttc, upper)) - indef(0)
    return out + np.maximum(upper - ttc, 0)

def p_cin_of(dur, rel_sev, k):
    return logf2(rel_sev*dur, k)

def p_cancer_of(dur, rel_sev, k, transform_prob):
    sev = intlogf2(rel_sev*dur, k)
    return 1 - np.power(1 - transform_prob, sev**2)


# %% Genotype parameters (hpvsim/parameters.py:_GENOTYPE_DEFAULTS)

GENOTYPES = sc.objdict(
    hpv16 = dict(rel_beta=1.0,  dur_precin=(3.0, 9.0), dur_cin=(5.0, 20.0), k=0.30, transform_prob=2.0e-3, sero_prob=0.75),
    hpv18 = dict(rel_beta=0.75, dur_precin=(2.5, 9.0), dur_cin=(5.0, 20.0), k=0.25, transform_prob=2.0e-3, sero_prob=0.56),
    hi5   = dict(rel_beta=0.9,  dur_precin=(2.5, 9.0), dur_cin=(4.5, 20.0), k=0.20, transform_prob=1.5e-3, sero_prob=0.60),
    ohr   = dict(rel_beta=0.9,  dur_precin=(2.5, 9.0), dur_cin=(4.5, 20.0), k=0.20, transform_prob=1.5e-3, sero_prob=0.60),
)
SHARED = dict(beta=0.25, transm2f=2.0, transf2m=1.0, dur_cancer=(8.0, 3.0), dur_inf_male=(1.0, 1.0),
              age_risk=dict(age=30, age_end=50, risk=2), hpv_control_prob=0.1, hpv_reactivation=0.025)
HIV_EFFECTS = dict(rel_sus=2.2, rel_sev_lo=1.5, rel_sev_hi=1.2, rel_imm_lo=0.36, rel_imm_hi=0.76)

lognorm = lambda m, s: ss.lognorm_ex(mean=ss.years(m), std=ss.years(s))
is_female = lambda sim: sim.people.female
is_male = lambda sim: ~sim.people.female


def age_ramp(age, age_risk):
    """ hpvsim.hpv._age_risk_ramp: 1 below age, risk above age_end, linear between """
    frac = np.clip((age - age_risk['age'])/(age_risk['age_end'] - age_risk['age']), 0, 1)
    return 1 + (age_risk['risk'] - 1)*frac


# %% One disease per genotype

def make_hpv(g, init_prev=0.03):
    gp = GENOTYPES[g]
    sp = SHARED
    S, P, Pe, C, Ce, X, L, Z = ss.states('S P Pe C Ce X L Z')

    # Branch probabilities that depend on the duration just spent in P or C. Each reads the
    # per-agent entry times the engine keeps (ti_P, ti_Pe, ...), since the duration itself is not exposed.
    def rel_sev(sim, uids):
        out = sim.connectors.cross.rel_sev[uids]
        hiv = sim.diseases.get('hiv')
        if hiv is not None:
            out = out*np.where(hiv.D[uids], HIV_EFFECTS['rel_sev_lo'], np.where(hiv.B[uids], HIV_EFFECTS['rel_sev_hi'], 1.0))
        return out

    def pcin(sim, uids):
        hpv = sim.diseases[g]
        dur = (hpv.ti_Pe[uids] - hpv.ti_P[uids])*sim.t.dt_year
        return p_cin_of(dur, rel_sev(sim, uids), gp['k'])

    def pcancer(sim, uids):
        hpv = sim.diseases[g]
        dur = (hpv.ti_Ce[uids] - hpv.ti_C[uids])*sim.t.dt_year
        return p_cancer_of(dur, rel_sev(sim, uids), gp['k'], gp['transform_prob'])

    ctrl = sp['hpv_control_prob']
    other_cancer = lambda sim: np.any([d.X.raw for name,d in sim.diseases.items() if name in GENOTYPES and name != g], axis=0)

    base = sp['beta']*gp['rel_beta']
    beta = ss.par(dict(mfnet=[base*sp['transm2f'], base*sp['transf2m']]), name='beta') # Per act; [m->f, f->m]

    hpv = ss.Disease(
        infection     = S >> P @ beta * (P + C),
        precin        = ss.Transition(P, Pe, dur=lognorm(*gp['dur_precin']), where=is_female),
        clear_male    = ss.Transition(P, S,  dur=lognorm(*sp['dur_inf_male']), where=is_male),
        to_cin        = ss.Transition(Pe, C, p=pcin),
        clear_precin  = ss.Transition(Pe, S, p=lambda sim, uids: (1 - pcin(sim, uids))*(1 - ctrl)),
        latent_precin = ss.Transition(Pe, L, p=lambda sim, uids: (1 - pcin(sim, uids))*ctrl),
        cin           = ss.Transition(C, Ce, dur=lognorm(*gp['dur_cin'])),
        to_cancer     = ss.Transition(Ce, X, p=pcancer),
        clear_cin     = ss.Transition(Ce, S, p=lambda sim, uids: (1 - pcancer(sim, uids))*(1 - ctrl)),
        latent_cin    = ss.Transition(Ce, L, p=lambda sim, uids: (1 - pcancer(sim, uids))*ctrl),
        reactivation  = L >> P @ ss.peryear(sp['hpv_reactivation']),
        cancer_death  = ss.Transition(X, ss.null, dur=lognorm(*sp['dur_cancer'])),
        cancelled     = ss.Transition([S, P, C, L], Z, rate=ss.peryear(1e3), where=other_cancer), # v3 _cancel_other_genotype_progression_for
        detected      = ss.outcome('to_cancer', delay=lognorm(1, 1)),
        init_prev     = init_prev,
        name          = g,
    )
    return hpv


# %% Cross-genotype immunity (v3 cross_genotype.CrossImmunity): per-agent continuous levels, so step() code

class CrossImmunity(ss.Connector):
    def __init__(self, genotypes, own_imm_hr=0.9, sus_med=0.3, sus_high=0.5, sev_med=0.5, sev_high=0.7, **kwargs):
        super().__init__(name='cross', **kwargs)
        self.genotypes = list(genotypes)
        def matrix(med, high):
            n = len(self.genotypes)
            m = np.full((n, n), med)
            for i,a in enumerate(self.genotypes):
                for j,b in enumerate(self.genotypes):
                    if i == j:
                        m[i, j] = 1.0 if a in ['hpv16', 'hpv18'] else own_imm_hr
                    elif {a, b} == {'hpv16', 'hpv18'}:
                        m[i, j] = high
            return m
        self.m_sus = matrix(sus_med, sus_high)
        self.m_sev = matrix(sev_med, sev_high)
        states = [ss.FloatArr('rel_sev', default=np.nan)]
        for g in self.genotypes:
            states += [ss.FloatArr(f'nab_{g}', default=0.0), ss.FloatArr(f'cell_{g}', default=0.0),
                       ss.FloatArr(f'sus_imm_{g}', default=0.0), ss.FloatArr(f'sev_imm_{g}', default=0.0)]
        self.define_states(*states)
        self.define_pars(
            rel_sev_dist = ss.normal(loc=1.0, scale=0.2),
            imm_init     = ss.uniform(low=0.5, high=0.95),
            cell_imm_init = ss.uniform(low=0.3, high=0.9),
            sero = ss.bernoulli(p=0.5),
        )
        return

    def init_post(self):
        super().init_post()
        self.ensure_rel_sev()
        return

    def ensure_rel_sev(self):
        uids = self.sim.people.auids
        new = uids[np.isnan(self.rel_sev[uids])]
        if len(new):
            self.rel_sev[new] = np.abs(self.pars.rel_sev_dist.rvs(new))
        return

    def step(self):
        sim = self.sim
        self.ensure_rel_sev()
        hiv = sim.diseases.get('hiv')
        for g in self.genotypes:
            hpv = sim.diseases[g]
            ev = hpv.engine.events # Engine internals: who went through each transition this step
            cleared = ss.uids.concatenate([ev.get('clear_precin', ss.uids()), ev.get('clear_cin', ss.uids())])
            if not len(cleared):
                continue
            nab = self.pars.imm_init.rvs(cleared)
            cell = self.pars.cell_imm_init.rvs(cleared)
            if hiv is not None:
                f = np.where(hiv.D[cleared], HIV_EFFECTS['rel_imm_lo'], np.where(hiv.B[cleared], HIV_EFFECTS['rel_imm_hi'], 1.0))
                nab, cell = nab*f, cell*f
            first = self[f'nab_{g}'][cleared] == 0
            self.pars.sero.set(p=GENOTYPES[g]['sero_prob'])
            sero = self.pars.sero.rvs(cleared) # One draw per agent; only used for first clearances
            nab = np.where(first, nab*sero, np.maximum(self[f'nab_{g}'][cleared], nab))
            cell = np.where(first, cell, np.maximum(self[f'cell_{g}'][cleared], cell))
            self[f'nab_{g}'][cleared] = nab
            self[f'cell_{g}'][cleared] = cell

        auids = sim.people.auids
        nab = np.column_stack([self[f'nab_{g}'][auids] for g in self.genotypes])
        cell = np.column_stack([self[f'cell_{g}'][auids] for g in self.genotypes])
        with np.errstate(all='ignore'): # Spurious BLAS warnings with NumPy 2.2 on macOS
            sus = np.clip(nab @ self.m_sus.T, 0, 1)
            sev = np.clip(cell @ self.m_sev.T, 0, 0.99)
        for i,g in enumerate(self.genotypes):
            self[f'sus_imm_{g}'][auids] = sus[:, i]
            self[f'sev_imm_{g}'][auids] = sev[:, i]
        return


def immunity_modifiers(genotypes):
    """ The connector holds the levels; these apply them, so the HPV modules' own rel_sus stays untouched """
    mods = []
    for g in genotypes:
        mods += [
            ss.modify(f'{g}.rel_sus', multiply=lambda sim, g=g: 1 - sim.connectors.cross[f'sus_imm_{g}'].raw, name=f'nab_{g}'),
            ss.modify(f'{g}.precin', multiply=lambda sim, g=g: 1/(1 - sim.connectors.cross[f'sev_imm_{g}'].raw), name=f'sevimm_{g}'), # v3: dur_precin*(1 - sev_imm)
            ss.modify(f'{g}.reactivation', multiply=lambda sim, g=g: 1 - sim.connectors.cross[f'sev_imm_{g}'].raw, name=f'react_{g}'),
            ss.modify(f'{g}.cin', multiply=lambda sim: 1/age_ramp(sim.people.age.raw, SHARED['age_risk']), name=f'agerisk_{g}'), # Rate multiplier, so 1/ramp on the duration
        ]
    return mods


# %% HIV: CD4 bands as states, so its effects on HPV are plain conditions

def make_hiv():
    S, A, B, D = ss.states('S A B D') # A: CD4>500, B: 200-500, D: <200
    T = ss.State('T', rel_trans=0.04) # On ART. (A + B + D + 0.04*T) with a dict beta raises TypeError: see NOTES
    hiv = ss.Disease(
        infection = S >> A @ ss.par(dict(mfnet=[0.03, 0.015]), name='beta') * (A + B + D + T),
        decline1  = A >> B @ ss.years(4),
        decline2  = B >> D @ ss.years(3),
        aids      = ss.Transition(D, ss.null, dur=ss.years(2)),
        art       = (A | B | D) >> T @ ss.timeseries({1990: 0, 2005: 0, 2015: 0.05}, mode='linear'),
        init_prev = 0.05,
        name      = 'hiv',
    )
    return hiv

def hiv_modifiers(genotypes, hiv):
    return [ss.modify(f'{g}.rel_sus', multiply=HIV_EFFECTS['rel_sus'], where=hiv.B | hiv.D, name=f'hiv_sus_{g}') for g in genotypes]


def make_population(birth_rate=0.025, death_rate=0.01):
    """ Births and deaths as a declared module: ss.birth() places newborns in the first state of every declared disease """
    N = ss.State('N')
    return ss.Disease(births=ss.birth('-> N', rate=ss.peryear(birth_rate)), deaths=ss.death('N ->', rate=ss.peryear(death_rate)), name='pop')


# %% Results that are not one disease's states or flows

WHO_EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 150])
WHO_W = np.array([8860, 8690, 8600, 8470, 8220, 7930, 7610, 7150, 6590, 6040, 5370, 4550, 3720, 2960, 2210, 1520, 910, 440, 150, 40, 5])

class HPVTotal(ss.Analyzer):
    """ Any-genotype prevalence and CIN by age, and cancers by WHO age band for ASR (v3 HPVTotal, cut down) """
    def __init__(self, genotypes, age_bins=(15, 25, 35, 45, 55, 65), **kwargs):
        super().__init__(name='total', **kwargs)
        self.genotypes = list(genotypes)
        self.age_bins = np.array(age_bins)
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('hpv_prev_f', dtype=float, scale=False, label='Any-genotype HPV prevalence, women 15-65'),
            ss.Result('cin_prev_f', dtype=float, scale=False, label='Any-genotype CIN prevalence, women 15-65'),
            ss.Result('new_cancers', dtype=float, label='New cancers (onset)'),
            ss.Result('asr_cancer', dtype=float, scale=False, label='ASR cancer incidence per 100k (WHO 2000)'),
        )
        self.cin_by_age = np.zeros((len(self.timevec), len(self.age_bins) - 1))
        self.cancers_by_band = np.zeros((len(self.timevec), len(WHO_W)))
        self.women_by_band = np.zeros((len(self.timevec), len(WHO_W)))
        return

    def step(self):
        sim = self.sim
        ti = self.ti
        ppl = sim.people
        uids = ppl.auids
        f = ppl.female[uids]
        age = ppl.age[uids]
        hpvs = [sim.diseases[g] for g in self.genotypes]
        inf = np.any([h.P[uids] | h.Pe[uids] | h.C[uids] | h.Ce[uids] for h in hpvs], axis=0)
        cin = np.any([h.C[uids] | h.Ce[uids] for h in hpvs], axis=0)
        adult = f & (age >= 15) & (age < 65)
        self.results.hpv_prev_f[ti] = inf[adult].mean()
        self.results.cin_prev_f[ti] = cin[adult].mean()
        for b in range(len(self.age_bins) - 1):
            inb = f & (age >= self.age_bins[b]) & (age < self.age_bins[b+1])
            self.cin_by_age[ti, b] = cin[inb].mean() if inb.any() else np.nan
        new = ss.uids.concatenate([h.engine.events.get('to_cancer', ss.uids()) for h in hpvs])
        cancer_ages = ppl.age[new]
        self.results.new_cancers[ti] = len(new)*sim.current_scale
        self.cancers_by_band[ti] = np.histogram(cancer_ages, bins=WHO_EDGES)[0]
        self.women_by_band[ti] = np.histogram(age[f], bins=WHO_EDGES)[0]
        rates = np.divide(self.cancers_by_band[ti]/sim.t.dt_year, self.women_by_band[ti], out=np.zeros(len(WHO_W)), where=self.women_by_band[ti] > 0)*1e5
        self.results.asr_cancer[ti] = (rates*WHO_W).sum()/WHO_W.sum()
        return


# %% Build and run

def make_sim(genotypes=GENOTYPES.keys(), n_agents=5000, start=1980, stop=2030, hiv=True, **kwargs):
    genotypes = list(genotypes)
    hpvs = [make_hpv(g) for g in genotypes]
    diseases = [make_population()] + list(hpvs)
    connectors = [CrossImmunity(genotypes)] + immunity_modifiers(genotypes)
    if hiv:
        h = make_hiv()
        diseases.append(h)
        connectors += hiv_modifiers(genotypes, h)
    net = ss.MFNet(n_contacts=ss.poisson(0.7), concurrency=0.15, duration=ss.lognorm_ex(mean=ss.years(2), std=ss.years(3)),
                   acts=ss.poisson(lam=ss.freqperyear(60)), debut=ss.normal(loc=16, scale=2), name='mfnet')
    sim = ss.Sim(
        diseases     = diseases,
        networks     = net,
        connectors   = connectors,
        analyzers    = HPVTotal(genotypes),
        demographics = None, # ss.Births() adds agents that are in no state of any declared disease: see NOTES
        people       = ss.People(n_agents, age_data=ss.uniform(0, 70)),
        start=start, stop=stop, dt=ss.years(0.25), rand_seed=1, verbose=0,
        stratify     = dict(age=[0, 15, 25, 35, 45, 55, 65]),
        **kwargs,
    )
    return sim


def report(sim):
    res = sim.results
    tot = res.total
    years = np.asarray(sim.t.yearvec, dtype=float)
    yi = lambda y: int(np.argmin(np.abs(years - y)))
    print(f'\n{sim.solver} | {len(sim.people)} agents alive at end | {sim.timer.elapsed if hasattr(sim, "timer") else ""}')
    print(f'{"year":>6} {"HPV prev F15-65":>16} {"CIN prev F15-65":>16}')
    for y in [1980, 1990, 2000, 2010, 2020, 2030]:
        if years[0] <= y <= years[-1]:
            i = yi(y)
            print(f'{y:>6} {tot.hpv_prev_f[i]:>16.3f} {tot.cin_prev_f[i]:>16.4f}')
    print('Infections by genotype (cumulative):', {g: int(res[g].cum_infection[-1]) for g in GENOTYPES if g in res})
    print('Prevalent infection by genotype at end (n_P):', {g: int(res[g].n_P[-1]) for g in GENOTYPES if g in res})
    print('CIN onsets / cancers / cancer deaths / reactivations by genotype:')
    for g in GENOTYPES:
        if g in res:
            r = res[g]
            print(f'  {g:6} {int(r.to_cin.sum()):5} {int(r.to_cancer.sum()):4} {int(r.cancer_death.sum()):4} {int(r.reactivation.sum()):4}')
    an = sim.analyzers.total
    late = years >= years[-1] - 20
    cin_age = np.nanmean(an.cin_by_age[late], axis=0)
    print('CIN prevalence by age, last 20y (women):', dict(zip([f'{a}-{b}' for a,b in zip(an.age_bins[:-1], an.age_bins[1:])], np.round(cin_age, 4))))
    canc = an.cancers_by_band[late].sum(0)/(len(years[late])*sim.t.dt_year)
    pop = an.women_by_band[late].mean(0)
    rates = np.divide(canc, pop, out=np.zeros_like(canc), where=pop > 0)*1e5
    print(f'ASR cancer incidence, last 20y pooled: {(rates*WHO_W).sum()/WHO_W.sum():.1f} per 100k (crude {canc.sum()/pop.sum()*1e5:.1f}); {int(an.cancers_by_band[late].sum())} cancers')
    print('Automatic, by age (sim stratified by age): hpv16 infections (cum):', np.round(res.hpv16.infection.by('age').values.sum(0)).astype(int).tolist())
    print('                                           hpv16 n_C at end:', res.hpv16.n_C.by('age').values[-1].astype(int).tolist())
    print('hpv16 detected-cancer outcome total:', res.hpv16.detected.sum() if 'detected' in res.hpv16 else 'n/a')
    ppl = sim.people
    u = ppl.auids
    fem = ppl.female[u] & (ppl.age[u] >= 15) & (ppl.age[u] < 65)
    n_gt = np.sum([sim.diseases[g].P[u] | sim.diseases[g].C[u] for g in GENOTYPES], axis=0)
    print(f'Women 15-65 with HPV at end: {np.sum(fem & (n_gt > 0))}, of whom with 2+ genotypes at once: {np.sum(fem & (n_gt > 1))}')
    if 'hiv' in res:
        hv = sim.diseases.hiv
        pos = (hv.A[u] | hv.B[u] | hv.D[u] | hv.T[u])
        low = hv.B[u] | hv.D[u]
        print(f'HPV prevalence at end, women 15-65: HIV- {np.mean(n_gt[fem & ~pos] > 0):.3f}, HIV+ CD4<500 {np.mean(n_gt[fem & low] > 0):.3f} (n={np.sum(fem & low)})')
        h = res.hiv
        n_inf = h.n_A + h.n_B + h.n_D + h.n_T
        print(f'HIV prevalence 2000/2030: {n_inf[yi(2000)]/res.n_alive[yi(2000)]:.3f} / {n_inf[-1]/res.n_alive[-1]:.3f}; on ART at end: {h.n_T[-1]/n_inf[-1]:.2f}')
    return


# %% Alternatives that were tried: genotypes as a strain dimension of one disease

def strain_alternatives(n_agents=3000):
    """ What v4's strain dimension can and cannot express for HPV genotypes """
    G = list(GENOTYPES.keys())
    out = sc.objdict()
    net = lambda: ss.MFNet(n_contacts=ss.poisson(0.7), concurrency=0.15, duration=ss.lognorm_ex(mean=ss.years(2), std=ss.years(3)), name='mfnet')
    kw = dict(n_agents=n_agents, start=1990, stop=2000, dt=ss.years(0.25), verbose=0)
    def base(superinf=False):
        S, P, C, R = ss.states('S P C R')
        beta = ss.par(dict(mfnet=[0.5, 0.25]), name='beta')
        extra = dict(superinfection=P >> P @ ss.par(dict(mfnet=[0.5, 0.25]), name='beta_super') * (P + C)) if superinf else {}
        return ss.Disease(infection=S >> P @ beta * (P + C), to_cin=P >> C @ ss.years(3), clear=P >> R @ ss.years(1),
                          regress=C >> R @ ss.years(5), waning=R >> S @ ss.years(20), init_prev=0.1, name='hpv', **extra)
    imm = np.full((4, 4), 0.3); np.fill_diagonal(imm, 0.9)
    trials = dict(
        strain_immunity_mfnet = lambda: ss.Sim(base().stratify(genotype=G, immunity=imm), networks=net(), **kw),
        strain_mfnet          = lambda: ss.Sim(base().stratify(genotype=G), networks=net(), **kw),
        carriage_15_levels    = lambda: ss.Sim(base(True).stratify(genotype=['+'.join(c) for k in range(1, 5) for c in itertools.combinations(G, k)]), networks=net(), **kw),
    )
    for key,make in trials.items():
        try:
            sim = make().run()
            out[key] = f'ran; n_P by level at end: {sim.results.hpv.n_P.by("genotype").values[-1].astype(int).tolist()}'
        except Exception as e:
            out[key] = f'{type(e).__name__}: {str(e).splitlines()[0][:160]}'
    for k,v in out.items():
        print(f'  {k}: {v}')
    return out


def check_sex_dimension():
    """ stratify(sex=...) in an agent run: does it use people.female? """
    S, I = ss.states('S I')
    d = ss.Disease(infection=S >> I @ 0.1 * I, recovery=I >> S @ ss.years(1), init_prev=0.1, name='d')
    out = {}
    for key,levels in [('sex', ['female', 'male']), ('female', [False, True])]:
        try:
            sim = ss.Sim(sc.dcp(d), n_agents=2000, start=2000, stop=2002, verbose=0, stratify={key: levels}).run()
            ppl = sim.people
            attr = ppl[key][ppl.auids]
            fem = ppl.female[ppl.auids]
            out[key] = f'agreement of "{key}" level with people.female: {np.mean((attr == (0 if key == "sex" else 1)) == fem):.2f}'
        except Exception as e:
            out[key] = f'{type(e).__name__}: {str(e).splitlines()[0][:160]}'
    for k,v in out.items():
        print(f'  stratify({k}=...): {v}')
    return out


def check_junction_delay():
    """ How long people sit in a zero-dwell junction state before an immediate p= branch fires """
    S, P, Pe, C = ss.states('S P Pe C')
    d = ss.Disease(precin=P >> Pe @ ss.years(1), to_c=ss.Transition(Pe, C, p=0.5), back=ss.Transition(Pe, S, p=0.5),
                   init=dict(P=1.0), name='d')
    sim = ss.Sim(d, n_agents=2000, start=2000, stop=2004, dt=ss.years(0.25), verbose=0).run()
    dd = sim.diseases.d
    u = (dd.C | dd.S).uids
    lag = (np.where(dd.C[u], dd.ti_C[u], dd.ti_S[u]) - dd.ti_Pe[u])
    print(f'  steps spent in junction Pe before the p= branch: mean {np.nanmean(lag):.2f} (n={len(u)}), mean n_Pe {sim.results.d.n_Pe.mean():.1f}')
    return lag


if __name__ == '__main__':
    T = sc.timer()
    sim = make_sim()
    sim.run()
    report(sim)
    print(f'run time: {T.toc(output=True):.1f} s')

    print('\nStrain-dimension alternatives:')
    strain_alternatives()
    print('\nSex dimension check:')
    check_sex_dimension()
    print('\nJunction check:')
    check_junction_delay()
