"""
STIsim BV (CST model) + GUD + coinfection connectors, ported to Starsim v4.

v3 sources: stisim/diseases/bv.py (BV), stisim/interventions/bv_interventions.py (treat_BV),
stisim/diseases/gud.py (GUD), stisim/connectors/hiv_sti.py, stisim/connectors/gud_syph.py.
HIV, syphilis, CT/NG/TV are deliberately minimal: just enough states for the connectors.

Run: conda run -n starsim python port.py
"""
import numpy as np
import sciris as sc
import starsim as ss

ppl = ss.People(1)  # Only so that people's attributes can be used in lazy conditions before the sim exists
adult_f = ppl.female & (ppl.age >= 15)
by_stable = lambda *v: ss.by(stable=list(v))  # A value for each stable CST (1, 3, 4)


#%% BV: vaginal microbiome as community state types (v3: stisim/diseases/bv.py:369-770)

def recur_rate(sim, uids):
    """ Recurrence after treatment: a hazard that decays with time since treatment (agents only) """
    bv = sim.diseases.bv
    months = (sim.ti - bv.ti_P[uids]) * sim.t.dt.months
    return ss.permonth(bv.pars.recur0 * np.exp(-months / bv.pars.recur_tau))


class BV(ss.Disease):
    """
    CST1 (Lactobacillus crispatus), CST3 (L. iners), CST4 (BV: anaerobes), for women 15+.
    U holds everyone without a CST (men, girls); P is the post-treatment window.
    Each woman has a fixed 'stable' CST (a stratification), which sets her risk of
    worsening, her chance of symptoms, and the CST she returns to after clearance.
    """
    U, CST1, CST3, CST4a, CST4p, CST4s, P = ss.states('U CST1 CST3 CST4a CST4p CST4s P')

    # Entering the microbiome model at 15, at the stable CST (v3: set_cst(), init_post())
    debut = ss.progression('U -> CST1 | CST3 | CST4p | CST4a', rate=ss.perday(1), where=adult_f,
                           p=[by_stable(1, 0, 0), by_stable(0, 1, 0), by_stable(0, 0, 0.1), None])

    # Spontaneous worsening (v3: change_cst(), p_cst_change * rel_sus)
    shift13 = ss.progression('CST1 -> CST3', rate=ss.probpermonth(0.10))
    onset = ss.progression('CST3 -> CST4p | CST4a', rate=ss.probpermonth(0.05), p=[by_stable(0.8, 0.7, 0.1), None])

    # Transient CST3 in stable-CST1 women returns to CST1 (v3: set_duration(cst=3), clear_infection())
    revert31 = ss.progression('CST3 -> CST1', dur=ss.uniform(ss.weeks(1), ss.weeks(18)), where=ss.Cond('dim', 'stable') == '1')

    # BV: presymptomatic -> symptomatic; spontaneous clearance to CST1 or CST3 by stable CST (v3: spontaneous_clearance)
    symptoms = CST4p >> CST4s @ ss.uniform(ss.weeks(1), ss.weeks(2))
    clear_s = ss.progression('CST4s -> CST1 | CST3', dur=ss.uniform(ss.weeks(5), ss.weeks(50)), p=[by_stable(1, 0, 0), None])
    clear_a = ss.progression('CST4a -> CST1 | CST3', dur=ss.uniform(ss.weeks(5), ss.weeks(50)), p=[by_stable(1, 0, 0), None])

    # Treatment: moved by ss.treat() along tx; never happens on its own
    tx = (CST4a | CST4p | CST4s) >> P @ 0

    # Post-treatment window: recurrence (time-since-treatment hazard) competes with settling back
    recur = ss.progression('P -> CST4p | CST4a', rate=recur_rate, p=[by_stable(0.8, 0.7, 0.1), None])
    settle = ss.progression('P -> CST1 | CST3', dur=ss.months(6), p=[by_stable(1, 0, 0), None])

    # Results that are not counts (v3: cst4_prevalence etc., among women 15+). In the class body these are
    # silently dropped (ss.transitions.class_members() keeps only State/Transition/par/outcome), so they go in __init__
    # prevalence = (CST4a | CST4p | CST4s) / (CST1 | CST3 | CST4a | CST4p | CST4s | P)

    def __init__(self, **kwargs):
        c = type(self)
        women = c.CST1 | c.CST3 | c.CST4a | c.CST4p | c.CST4s | c.P
        # A ratio result raises ZeroDivisionError on a step where the denominator is 0 (here t=0, before anyone has
        # debuted), so only the sums are declared and the ratios are taken afterwards (see bv_prev())
        # super().__init__(prevalence=(c.CST4a | c.CST4p | c.CST4s) / women, symp_prevalence=c.CST4s / women, **kwargs)
        # A set of states (|) passed as a result is silently dropped too; a sum (+) is kept
        super().__init__(n_bv=c.CST4a + c.CST4p + c.CST4s, n_women=c.CST1 + c.CST3 + c.CST4a + c.CST4p + c.CST4s + c.P, **kwargs)
        self.define_pars(recur0=0.06, recur_tau=3.0, p_circumcised=0.4)  # Recurrence hazard at treatment (per month), its decay (months)
        self.define_states(
            ss.BoolArr('poor_hygiene', default=ss.bernoulli(0.55)),   # v3: p_poor_menstrual_hygiene
            ss.BoolArr('circumcised', default=ss.bernoulli(0.4)),     # v3: p_circumcised (men; women ignore it)
        )
        return


#%% Minimal HIV with a continuous CD4 (others port the full model)

class HIV(ss.Disease):
    S = ss.State()
    A = ss.State(rel_trans=5)    # Acute
    C = ss.State()               # Chronic
    D = ss.State(rel_trans=2)    # AIDS
    infection = S >> A @ dict(mfnet=[0.0015, 0.0008]) * (A + C + D)
    acute = A >> C @ ss.months(3)
    progress = C >> D @ ss.years(8)
    death = D >> ss.null @ ss.years(2)

    def __init__(self, **kwargs):
        super().__init__(init_prev=0.05, **kwargs)
        self.define_states(ss.FloatArr('cd4', default=np.nan, label='CD4 count'))
        return

    def step_state(self):
        """ CD4 as a function of time in each state: 800 -> 500 over acute, 500 -> 200 over chronic, then halving yearly """
        super().step_state()
        ti, dt = self.ti, self.t.dt.years
        cd4 = np.full(len(self.cd4.raw), np.nan)
        for name, start, end, dur in [('A', 800, 500, 0.25), ('C', 500, 200, 8.0)]:
            inds = self[name].raw
            yrs = (ti - self[f'ti_{name}'].raw[inds]) * dt
            cd4[inds] = np.maximum(start - (start - end) * yrs / dur, end)
        inds = self.D.raw
        cd4[inds] = 200 * 0.5 ** ((ti - self.ti_D.raw[inds]) * dt)
        self.cd4.raw[:] = cd4
        return


#%% Minimal syphilis

class Syph(ss.Disease):
    S = ss.State()
    P = ss.State()                 # Primary
    Sec = ss.State()               # Secondary
    L = ss.State(rel_trans=0.2)    # Latent
    T = ss.State(rel_trans=0)      # Tertiary
    infection = S >> P @ dict(mfnet=[0.04, 0.03]) * (P + Sec + L)
    secondary = P >> Sec @ ss.weeks(6)
    latent = Sec >> L @ ss.months(3.6)
    tertiary = L >> T @ ss.years(20)
    treated = (P | Sec | L) >> S @ ss.peryear(0.3)  # Background treatment, so it stays endemic

    def __init__(self, **kwargs):
        super().__init__(init_prev=0.05, **kwargs)


#%% Minimal SEIS for CT/NG/TV (v3: stisim/diseases/sti.py SEIS), and GUD (v3: stisim/diseases/gud.py:97)

def seis(name, beta, dur_inf, init_prev):
    S, E, I = ss.states('S E I')
    return ss.Disease(
        infection = S >> E @ dict(mfnet=beta) * I,
        latent = E >> I @ ss.weeks(1),
        clearance = I >> S @ dur_inf,
        init_prev = init_prev,
        name = name,
    )


def make_gud():
    """ v3 GUD: SIS, lognormal duration (mean 3 months, sd 1); beta=1 there is a placeholder """
    return ss.Disease(
        infection = ss.transmission('S -> I', beta=dict(mfnet=[0.4, 0.4])),
        recovery = ss.progression('I -> S', dur=ss.lognorm_ex(mean=ss.months(3), std=ss.months(1))),
        init_prev = 0.02,
        name = 'gud',
    )


#%% Risk factors for BV: modifiers on the spontaneous transitions (v3: BV.set_rel_sus(), compute_circumcision_impact())

def concurrent(sim):
    """ Women with two or more current partners in the sexual network (v3: count_partners=True) """
    net = sim.networks.mfnet
    out = np.zeros(len(sim.people.uid.raw), dtype=bool)
    auids = sim.people.auids
    out[auids] = net.degrees(auids) > 1
    return out


def uncirc_partner(sim):
    """ Women with an uncircumcised male partner (v3: rr_uncircumcised, CST3 only) """
    net, bv = sim.networks.mfnet, sim.diseases.bv
    out = np.zeros(len(sim.people.uid.raw), dtype=bool)
    p1, p2 = net.edges.p1, net.edges.p2   # MFNet: p1 male, p2 female
    out[p2[~bv.circumcised.raw[p1]]] = True
    return out


def bv_risk(bv, rr_hygiene=4.1, rr_concurrency=1.28, rr_uncirc=4.0):
    """ One modifier per risk factor per transition; they multiply """
    mods = []
    for tr in [bv.shift13, bv.onset, bv.recur]:
        mods += [
            ss.modify(tr, multiply=rr_hygiene, where=bv.poor_hygiene, name=f'hygiene_{tr.name}'),
            ss.modify(tr, multiply=rr_concurrency, where=concurrent, name=f'concurrency_{tr.name}'),
        ]
    for tr in [bv.onset, bv.recur]: # CST3 -> CST4 only
        mods.append(ss.modify(tr, multiply=rr_uncirc, where=uncirc_partner, name=f'uncirc_{tr.name}'))
    return mods


#%% Connectors (v3: stisim/connectors/hiv_sti.py, gud_syph.py)

def cd4_mult(sim):
    """ A multiplier that is a continuous function of CD4: 1 at CD4 >= 500, rising linearly to 2 at CD4 = 0 """
    cd4 = sim.diseases.hiv.cd4.raw
    return np.where(np.isnan(cd4), 1.0, 1 + np.clip((500 - cd4) / 500, 0, 1))


def connectors(hiv, syph, ct, ng, tv, bv, gud, p=None):
    p = sc.mergedicts(dict(
        rel_sus_hiv_syph=2.67, rel_trans_hiv_syph=1.2,   # hiv_syph
        rel_sus_syph_hiv=1.5, rel_sus_syph_aids=1.5,     # v3 defaults are 1; non-1 here so they can be checked
        rel_trans_syph_hiv=1.2, rel_trans_syph_aids=1.2,
        rel_sus_hiv_tv=1.5,                              # hiv_tv
        rel_sus_hiv_ng=1.2, rel_trans_hiv_ng=1.2,        # hiv_ng
        rel_sus_hiv_ct=1.3,                              # hiv_ct (v3 default 1)
        rel_sus_hiv_bv=2.0, rel_trans_hiv_bv=2.0,        # hiv_bv
        rel_sus_syph_gud=2.0, rel_trans_syph_gud=2.0,    # gud_syph
        rel_sus_gud_syph=2.0, rel_trans_gud_syph=2.0,
        tertiary_aids=3.0,                               # Not in v3: faster progression of latent syphilis with CD4 < 200
    ), p)
    syph_active = syph.P | syph.Sec       # v3: syphilis.symptomatic
    bv_cst4 = bv.CST4a | bv.CST4p | bv.CST4s
    # The spec's lazy form, hiv.cd4 < 500, warns "non-initialized Arr" and loses its module: the condition prints as
    # "(cd4 < 500)" and is looked up in the *target's* module, so it fails with 'Syph object has no attribute "cd4"'.
    # low_cd4, aids = hiv.cd4 < 500, hiv.cd4 < 200
    low_cd4 = lambda sim: sim.diseases.hiv.cd4.raw < 500
    aids = lambda sim: sim.diseases.hiv.cd4.raw < 200
    m = ss.modify
    return [
        # HIV <-> syphilis
        m(hiv.rel_sus,   multiply=p['rel_sus_hiv_syph'],    where=syph_active, name='hiv_sus_syph'),
        m(hiv.rel_trans, multiply=p['rel_trans_hiv_syph'],  where=syph_active, name='hiv_trans_syph'),
        m(syph.rel_sus,  multiply=p['rel_sus_syph_hiv'],    where=low_cd4,     name='syph_sus_hiv'),
        m(syph.rel_sus,  multiply=p['rel_sus_syph_aids'],   where=aids,        name='syph_sus_aids'),
        m(syph.rel_trans, multiply=p['rel_trans_syph_hiv'], where=low_cd4,     name='syph_trans_hiv'),
        m(syph.rel_trans, multiply=p['rel_trans_syph_aids'], where=aids,       name='syph_trans_aids'),
        m(syph.tertiary, multiply=p['tertiary_aids'],       where=aids,        name='syph_tertiary_aids'),
        # HIV <- CT/NG/TV/BV
        m(hiv.rel_sus,   multiply=p['rel_sus_hiv_tv'],      where=tv.I,        name='hiv_sus_tv'),
        m(hiv.rel_sus,   multiply=p['rel_sus_hiv_ng'],      where=ng.I,        name='hiv_sus_ng'),
        m(hiv.rel_trans, multiply=p['rel_trans_hiv_ng'],    where=ng.I,        name='hiv_trans_ng'),
        m(hiv.rel_sus,   multiply=p['rel_sus_hiv_ct'],      where=ct.I,        name='hiv_sus_ct'),
        m(hiv.rel_sus,   multiply=p['rel_sus_hiv_bv'],      where=bv_cst4,     name='hiv_sus_bv'),
        m(hiv.rel_trans, multiply=p['rel_trans_hiv_bv'],    where=bv_cst4,     name='hiv_trans_bv'),
        # GUD <-> syphilis
        m(syph.rel_sus,  multiply=p['rel_sus_syph_gud'],    where=gud.I,       name='syph_sus_gud'),
        m(syph.rel_trans, multiply=p['rel_trans_syph_gud'], where=gud.I,       name='syph_trans_gud'),
        m(gud.rel_sus,   multiply=p['rel_sus_gud_syph'],    where=syph_active, name='gud_sus_syph'),
        m(gud.rel_trans, multiply=p['rel_trans_gud_syph'],  where=syph_active, name='gud_trans_syph'),
    ], p


#%% Workaround: people born through ss.Births are in no state of any declared disease (see NOTES.md, bug 1)

class place_newborns(ss.Connector):
    """ Put agents who are in no state of a declared disease into its first state, and draw their level of each dimension (bug 2) """
    def __init__(self, splits=None, **kwargs):
        super().__init__(**kwargs)
        self.splits = splits or {}
        for dim in self.splits:
            setattr(self, f'rng_{dim}', ss.random())
        return

    def step(self):
        sim = self.sim
        alive = sim.people.alive.raw
        new = None
        for dis in sim.diseases.values():
            states = dis.graph.people_states
            somewhere = np.zeros(len(alive), dtype=bool)
            for s in states:
                somewhere |= dis[s.name].raw
            new = ss.uids(np.flatnonzero(alive & ~somewhere))
            if len(new):
                first = states[0].name
                dis[first][new] = True
                dis[f'ti_{first}'][new] = self.ti
        if new is not None and len(new) and self.splits:
            for dim, split in self.splits.items():
                u = getattr(self, f'rng_{dim}').rvs(new)
                sim.people[dim][new] = np.searchsorted(np.cumsum(split), u, side='right').astype(float)
            sim.groups_changed(new)
        return


#%% Build and run

def make_sim(coinfect=True, treat=True, risk=True, cd4_continuous=False, seed=1, n_agents=5000):
    hiv = HIV(name='hiv')
    syph = Syph(name='syph')
    ct = seis('ct', [0.08, 0.06], ss.months(12), 0.05)
    ng = seis('ng', [0.10, 0.08], ss.months(6), 0.03)
    tv = seis('tv', [0.04, 0.02], ss.months(24), 0.05)
    gud = make_gud()
    bv = BV(name='bv')
    bv.stratify(stable=['1', '3', '4'], split=[0.10, 0.30, 0.60],
                adjust={'shift13': [0.1, 1, 2], 'onset': [0.1, 1, 2], 'recur': [0.1, 1, 2]})   # v3: rr_stable_cst*
    bv.stratify(douching=[False, True], split=[0.36, 0.64],
                adjust={'shift13': [1, 1.21], 'onset': [1, 1.21], 'recur': [1, 1.21]})        # v3: rr_douching
    bv.stratify(hc=[False, True], split=[0.8, 0.2], adjust={'onset': [1, 0.7], 'recur': [1, 0.7]},
                start_hc=ss.transfer(hc=True, rate=ss.peryear(0.1), states=['CST1', 'CST3', 'CST4a', 'CST4p', 'CST4s', 'P']),             # Not in v3: hormonal
                stop_hc=ss.transfer(hc=False, rate=ss.peryear(0.3)))              # contraception, time-varying

    conns, p = connectors(hiv, syph, ct, ng, tv, bv, gud) if coinfect else ([], None)
    conns = [place_newborns(splits=dict(stable=[0.10, 0.30, 0.60], douching=[0.36, 0.64], hc=[0.8, 0.2]))] + conns
    if coinfect and cd4_continuous: # Replace the two step-function CD4 effects on syphilis susceptibility by a continuous one
        conns = [c for c in conns if c.name not in ['syph_sus_hiv', 'syph_sus_aids']]
        conns.append(ss.modify(syph.rel_sus, multiply=cd4_mult, name='syph_sus_cd4'))
    if risk:
        conns += bv_risk(bv)

    intvs = []
    if treat: # v3: treat_BV: newly symptomatic, 90% seek care, efficacy by stable CST
        mtz = ss.Treatment(efficacy=by_stable(0.95, 0.90, 0.90), transition='tx')
        intvs.append(ss.treat(mtz, on=bv.symptoms, coverage=0.9, disease='bv', name='treat_bv'))

    net = ss.MFNet(n_contacts=ss.poisson(0.8), concurrency=0.15, duration=ss.lognorm_ex(mean=ss.years(2), std=ss.years(1)), name='mfnet')
    sim = ss.Sim(
        diseases=[hiv, syph, ct, ng, tv, gud, bv], networks=net, connectors=conns, interventions=intvs,
        demographics=[ss.Births(birth_rate=25), ss.Deaths(death_rate=10)],
        n_agents=n_agents, start=2000, stop=2015, dt=ss.months(1), rand_seed=seed, verbose=0,
    )
    return sim, p


def check_composition(sim, p):
    """ Does hiv.rel_sus multiply correctly over every combination of coinfections present? """
    d = sim.diseases
    for eng in [d.hiv.engine, d.syph.engine]:
        eng.cache.clear()   # The effective value is cached per step; recompute it for the states as they are now
    alive = sim.people.alive.raw
    syph_act = d.syph.P.raw | d.syph.Sec.raw
    cst4 = d.bv.CST4a.raw | d.bv.CST4p.raw | d.bv.CST4s.raw
    flags = sc.objdict(syph=syph_act, tv=d.tv.I.raw, ng=d.ng.I.raw, ct=d.ct.I.raw, bv=cst4)
    mults = dict(syph=p['rel_sus_hiv_syph'], tv=p['rel_sus_hiv_tv'], ng=p['rel_sus_hiv_ng'], ct=p['rel_sus_hiv_ct'], bv=p['rel_sus_hiv_bv'])
    expected = np.ones(len(alive))
    for k, f in flags.items():
        expected *= np.where(f, mults[k], 1.0)
    actual = d.hiv.rel_sus.effective.raw
    ok = np.allclose(actual[alive], expected[alive])
    n_co = np.sum([f[alive] for f in flags.values()], axis=0)
    combos = {int(k): int((n_co == k).sum()) for k in range(6)}
    print(f'  hiv.rel_sus composition matches product of 5 STI multipliers: {ok}; agents by number of coinfections: {combos}')

    cd4 = d.hiv.cd4.raw
    exp_syph = np.where(cd4 < 500, p['rel_sus_syph_hiv'], 1) * np.where(cd4 < 200, p['rel_sus_syph_aids'], 1) * np.where(d.gud.I.raw, p['rel_sus_syph_gud'], 1)
    ok2 = np.allclose(d.syph.rel_sus.effective.raw[alive], exp_syph[alive])
    print(f'  syph.rel_sus = CD4<500 x CD4<200 x GUD: {ok2}; n with CD4<500: {int(np.nansum(cd4[alive] < 500))}, <200: {int(np.nansum(cd4[alive] < 200))}')
    return ok and ok2


def bv_prev(r):
    return np.divide(r.bv.n_bv, r.bv.n_women, out=np.full(len(r.bv.n_bv), np.nan), where=r.bv.n_women > 0)


if __name__ == '__main__':
    T = sc.timer()
    sim, p = make_sim()
    sim.run()
    T.toc('Main sim')
    r = sim.results
    print(f'Solver: {sim.solver} ({sim.solver_reason}); unit: {sim.unit}')
    i5, i15 = 12*5, -1
    prev = bv_prev(r)
    print(f'BV (CST4) prevalence among women 15+: {prev[1]:.2f} -> {prev[i5]:.2f} (yr 5) -> {prev[i15]:.2f} (yr 15); symptomatic {r.bv.n_CST4s[-1]/r.bv.n_women[-1]:.3f}')
    n_w = sum(r.bv[f'n_{s}'][-1] for s in ['CST1', 'CST3', 'CST4a', 'CST4p', 'CST4s', 'P'])
    print('CST shares at end: ' + ', '.join(f'{s} {r.bv[f"n_{s}"][-1]/n_w:.2f}' for s in ['CST1', 'CST3', 'CST4a', 'CST4p', 'CST4s', 'P']))
    print(f'BV treated: {r.treat_bv.delivered.sum():.0f} (MTZ doses {14*r.treat_bv.delivered.sum():.0f}); cured into P: {r.bv.tx.sum():.0f}; recurred: {r.bv.recur.sum():.0f}; settled: {r.bv.settle.sum():.0f}; recurrence share {r.bv.recur.sum()/(r.bv.recur.sum()+r.bv.settle.sum()):.2f}')
    bv = sim.diseases.bv
    for k, cond in dict(poor_hygiene=bv.poor_hygiene.raw, ).items():
        women = ~bv.U.raw & sim.people.alive.raw
        cst4 = (bv.CST4a.raw | bv.CST4p.raw | bv.CST4s.raw)
        print(f'  BV prev with/without {k}: {cst4[women & cond].mean():.2f} / {cst4[women & ~cond].mean():.2f}')
    for dis in ['hiv', 'syph', 'ct', 'ng', 'tv', 'gud']:
        res = r[dis]
        states = [k[2:] for k in res.keys() if k.startswith('n_')]
        n = sum(res[f'n_{s}'][-1] for s in states)
        inf = n - res.n_S[-1]
        print(f'  {dis:5s} prevalence {inf/n:.3f}  cumulative infections {res.cum_infection[-1]:.0f}')

    print('Connector checks (at the end of the run):')
    check_composition(sim, p)
    print(sim.what_modifies('hiv.rel_sus', output=True))

    # Do the connectors change epidemiology?
    print('With vs without connectors (cumulative infections):')
    base, _ = make_sim(coinfect=False)
    base.run()
    for dis in ['hiv', 'syph', 'gud']:
        print(f'  {dis}: {r[dis].cum_infection[-1]:.0f} vs {base.results[dis].cum_infection[-1]:.0f}')
    print(f'  syph tertiary: {r.syph.cum_tertiary[-1]:.0f} vs {base.results.syph.cum_tertiary[-1]:.0f}')

    # A continuous CD4-dependent multiplier
    sim2, _ = make_sim(cd4_continuous=True)
    sim2.run()
    print(f'Continuous CD4 multiplier on syph.rel_sus: syph infections {sim2.results.syph.cum_infection[-1]:.0f}')
    d = sim2.diseases
    d.syph.engine.cache.clear()
    hivpos = ~np.isnan(d.hiv.cd4.raw) & sim2.people.alive.raw
    ok = np.allclose(d.syph.rel_sus.effective.raw[hivpos] / np.where(d.gud.I.raw[hivpos], 2.0, 1.0), cd4_mult(sim2)[hivpos])
    print(f'  effective syph.rel_sus among HIV+ = 1 + (500-CD4)/500 (x GUD): {ok}')

    # Treatment off: BV prevalence and recurrence
    sim3, _ = make_sim(treat=False)
    sim3.run()
    print(f'BV prevalence at end, with/without treatment: {bv_prev(r)[-1]:.3f} / {bv_prev(sim3.results)[-1]:.3f}')
    T.toc('All')
