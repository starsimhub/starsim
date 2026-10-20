"""
STIsim syphilis ported to the Starsim v4 declarative API.

v3 sources: stisim/diseases/syphilis.py, stisim/interventions/syphilis_interventions.py,
stisim/interventions/base_interventions.py (ANCTest, STITreatment, STIDx).

Run: conda run -n starsim python port.py
"""
import numpy as np
import sciris as sc
import starsim as ss
import stisim as sti  # Only for sti.StructuredSexual (a v3 network, run unchanged on v4)

ppl = ss.People(1)  # Needed only so that lazy conditions like ppl.age >= 0 can be written (see NOTES)

# %% Parameters (v3 SyphPars defaults unless noted)
dt = ss.months(1)
dt_yr = 1/12
BIG = 1e4  # A hazard per year big enough that "due" people leave this step: the idiom for an event at a known time

def per_act(p):
    """ v4 treats a bare beta as a hazard per unit time and converts it to 1-exp(-beta*dt) before the network sees it;
    StructuredSexual wants a per-act probability, so invert that conversion """
    return -np.log(1 - p)/dt_yr

beta_m2f    = 0.15                    # Per act (syph_dx_zim calibrated value)
rel_f2m     = 0.5                     # BaseSTIPars.rel_beta_f2m
beta_m2c    = 0.075*12                # v3: ss.permonth(0.075), as a rate per year
eff_condom  = 0.5
latent_hl   = 1.0                     # rel_trans_latent_half_life, years
p_symp_primary = dict(female=0.3, male=0.5)
birth_outcomes = dict( # nnd, stillborn, congenital (normal is the remainder; miscarriage is 0 everywhere in v3)
    active = [0.25, 0.25, 0.45],
    early  = [0.05, 0.10, 0.40],
    late   = [0.00, 0.10, 0.10],
)


# %% Functions used in rates and probabilities (agents only: they take uids)

def chancre_visible(sim, uids):
    """ Sex-specific probability that a primary chancre is at a visible site """
    female = sim.people.female[uids]
    return np.where(female, p_symp_primary['female'], p_symp_primary['male'])


def mother_stage(sim, uids):
    """ Mother's stage for each fetus: 0 = active (exposed/primary/secondary), 1 = early latent, 2 = late latent, -1 = not infectious """
    syph = sim.diseases.syph
    mom = np.asarray(sim.people.parent[uids])
    stage = np.full(len(uids), -1)
    ok = mom >= 0
    m = ss.uids(mom[ok])
    s = np.full(len(m), -1)
    s[np.asarray(syph.E[m] | syph.Pv[m] | syph.Ph[m] | syph.S2[m])] = 0
    s[np.asarray(syph.EL[m])] = 1
    s[np.asarray(syph.LL[m] | syph.T[m])] = 2
    stage[ok] = s
    return stage


def mtct_rate(sim, uids):
    """ Hazard of in-utero infection: v3 transmits with rel_trans=1 from every infectious stage, ignoring the latent decay """
    unborn = np.asarray(sim.people.age[uids] < 0)
    stage = mother_stage(sim, uids)
    syph = sim.diseases.syph
    mom = np.asarray(sim.people.parent[uids])
    ok = unborn & (stage >= 0)
    infectious = np.zeros(len(uids), dtype=bool)
    if ok.any():
        m = ss.uids(mom[ok])
        infectious[ok] = np.asarray(syph.Pv[m] | syph.Ph[m] | syph.S2[m] | syph.EL[m] | syph.LL[m]) # Not E or T (v3: infectious = symptomatic | latent)
    return beta_m2c*infectious


def outcome_p(i):
    """ Branch probability of congenital outcome i, from the mother's stage when the fetus was infected """
    table = np.array([birth_outcomes['active'], birth_outcomes['early'], birth_outcomes['late']])
    table = np.hstack([table, 1 - table.sum(axis=1, keepdims=True)]) # Normal is the remainder
    def p(sim, uids):
        stage = mother_stage(sim, uids)
        return np.where(stage >= 0, table[np.maximum(stage, 0), i], float(i == 3))
    p.__name__ = f'outcome_p{i}'
    return p


def before_delivery(sim, uids):
    """ Due one step before the mother delivers, so the fetal death is a fetal loss (v3 issue #588); or now, if already born
    (infected within a step of delivery: then it is a neonatal death, as in v3) """
    preg = sim.demographics.pregnancy
    mom = np.asarray(sim.people.parent[uids])
    ok = mom >= 0
    due = np.asarray(sim.people.age[uids] >= 0)
    ti_del = preg.ti_delivery[ss.uids(mom[ok])]
    due[ok] |= np.asarray(sim.ti >= ti_del - 1)
    return BIG*due


def fetal_eff(sim):
    """ Fetal treatment efficacy: reduced in the last trimester (v3 fetus_age_cutoff_treat_eff = -0.25) """
    return np.where(sim.people.age.raw > -0.25, 0.75, 0.98)


# %% The disease

class Syphilis(ss.Disease):
    # Susceptible: naive; previously infected and still treponemal-positive; previously infected and sero-reverted
    S, Sp, Sr = ss.states(3)
    # Adult natural history: exposed, primary (chancre visible / hidden), secondary, early latent, late latent, tertiary
    E, Pv, Ph, S2, EL, LL, T = ss.states(7)
    # Congenital: infected in utero (outcome not yet drawn), awaiting each outcome, born with congenital syphilis
    C, Cnnd, Cstill, Ccs, Cnorm, CS = ss.states(6)

    # Sexual transmission: stage weights here, latent decay per agent via a modifier of rel_trans (make_sim). E and T do not transmit.
    beta = dict(structuredsexual=[per_act(beta_m2f), per_act(beta_m2f*rel_f2m)], maternalnet=0)
    infection = (S | Sp | Sr) >> E @ beta * (Pv + Ph + S2 + EL + LL)

    # Natural history
    primary_v  = ss.Transition(E, Pv, p=chancre_visible, dur=ss.normal(ss.days(50), ss.days(10)))
    primary_h  = ss.Transition(E, Ph, dur=ss.normal(ss.days(50), ss.days(10))) # The remainder
    secondary  = ss.Transition([Pv, Ph], S2, dur=ss.normal(ss.weeks(6), ss.weeks(1)), move=dict(rpr=True)) # Non-treponemal titre rises by secondary
    latent     = S2 >> EL @ ss.lognorm_ex(ss.months(3.6), ss.months(1.5))
    reactivate = ss.Transition(EL, S2, p=0.35, dur=ss.lognorm_ex(ss.years(1), ss.years(1)), move=dict(rpr=True))
    to_late    = EL >> LL @ ss.uniform(ss.months(22), ss.months(24)) # Competes with reactivation: the earlier wins
    tertiary   = ss.Transition(LL, T, p=0.35, dur=ss.normal(ss.years(18), ss.years(2))) # v3: 20 y from latent onset
    stay_LL    = ss.Transition(LL, T, p=0.65, rate=ss.peryear(0)) # "The other 65% stay": p= must sum to 1 (E205), so a branch that never fires
    tert_death = ss.Transition(T, ss.null, p=0.05, dur=ss.lognorm_ex(ss.years(5), ss.years(5)))
    stay_T     = ss.Transition(T, ss.null, p=0.95, rate=ss.peryear(0))

    # Treatment: one transition out of every infected stage, which only interventions move people along
    cure = (E | Pv | Ph | S2 | EL | LL | T) >> Sp @ ss.peryear(0)
    trep_revert = ss.Transition(Sp, Sr, p=0.2, dur=ss.lognorm_ex(ss.years(10), ss.years(5))) # 20% sero-revert
    stay_Sp     = ss.Transition(Sp, Sr, p=0.8, rate=ss.peryear(0)) # ... the rest stay trep+ for life

    # Congenital: infection of the fetus, from the mother's state (not a network transmission: see NOTES)
    mtct    = S >> C @ mtct_rate
    c_nnd   = ss.Transition(C, Cnnd,   p=outcome_p(0))
    c_still = ss.Transition(C, Cstill, p=outcome_p(1))
    c_cs    = ss.Transition(C, Ccs,    p=outcome_p(2))
    # c_norm = ss.Transition(C, Cnorm) # The remainder, as the spec allows: raises E209 UndrivenTransition (see NOTES)
    c_norm  = ss.Transition(C, Cnorm,  p=outcome_p(3))
    stillbirth = Cstill >> ss.null @ before_delivery
    nnd        = ss.progression('Cnnd ->', rate=BIG, where=ppl.age >= 0)
    born_cs    = ss.progression('Ccs -> CS', rate=BIG, where=ppl.age >= 0)
    born_ok    = ss.progression('Cnorm -> S', rate=BIG, where=ppl.age >= 0) # v3 leaves these babies non-susceptible for life (see NOTES)
    fetal_cure = (Cnnd | Cstill | Ccs) >> Cnorm @ ss.peryear(0) # Treatment of the mother cures the fetus
    cs_cure    = CS >> S @ ss.peryear(0) # Newborn treatment

    def __init__(self, **kwargs):
        # Extra results: sums and ratios of states. In the class body they are silently ignored (class_members() collects
        # only states, transitions, pars, and outcomes), so they are passed as keywords here (see NOTES)
        c = type(self)
        infected = c.E + c.Pv + c.Ph + c.S2 + c.EL + c.LL + c.T
        super().__init__(infected=infected, ever=infected + c.Sp + c.Sr, prev=infected/(c.S + c.Sp + c.Sr + infected), **kwargs)
        return

    def step_state(self):
        self.place_newcomers()
        super().step_state()
        return

    def place_newcomers(self):
        """ Agents made by ss.Pregnancy/ss.Births are in no state of a declared disease: put them in S (see NOTES) """
        names = self.graph.people_states.keys() if isinstance(self.graph.people_states, dict) else [s.name for s in self.graph.people_states]
        anywhere = np.zeros(len(self.S.raw), dtype=bool)
        for name in names:
            anywhere |= self[name].raw
        auids = self.sim.people.auids
        new = auids[~anywhere[auids]]
        if len(new):
            self.S[new] = True
            self.ti_S[new] = self.ti
        return new


def latent_decay(sim):
    """ Latent infectiousness halves every latent_hl years since entering latent (v3 set_latent_trans). Relies on ti_EL
    being kept after people leave EL for LL. Returns one value per UID; the modifier's where= picks the latent """
    syph = sim.diseases.syph
    return np.exp(-np.log(2)/latent_hl*(sim.ti - syph.ti_EL.raw)*dt_yr)


# %% Interventions

class ANCVisits(ss.Intervention):
    """ One ANC visit per pregnancy, 1-8 months after conception, for visit_prob of pregnancies (v3 ANCTest scheduling) """
    def __init__(self, visit_prob=0.85, **kwargs):
        super().__init__(**kwargs)
        self.visit_prob = ss.bernoulli(p=visit_prob)
        self.timing = ss.randint(1, 9)
        self.define_states(ss.FloatArr('ti_visit'))
        return

    def step(self):
        preg = self.sim.demographics.pregnancy
        new = (preg.ti_pregnant == self.ti).uids
        go = self.visit_prob.filter(new)
        self.ti_visit[go] = self.ti + self.timing.rvs(go)
        return


def make_interventions(syph, capacity=None):
    bpg  = ss.Treatment(efficacy=0.98, transition='cure', name='bpg')    # Benzathine penicillin
    dual = ss.Test(sensitivity=dict(Sp=0.95, E=0.2, Pv=0.2, Ph=0.2, S2=0.95, EL=0.95, LL=0.95, T=0.95), specificity=0.99) # syph_dx_zim 'dual'; Sr not detected
    visits = ANCVisits(name='anc_visits')
    anc = ss.screen(dual, on=lambda sim: sim.interventions.anc_visits.ti_visit == sim.ti, on_positive=ss.treat(bpg, capacity=capacity), start=2005, name='anc')
    symp = ss.treat(bpg, eligible=syph.Pv | syph.S2, coverage=0.1, interval=dt, name='symp') # Care-seeking for visible symptoms
    fetal = ss.treat(ss.Treatment(efficacy=fetal_eff, transition='fetal_cure', name='fetal_tx'), on=syph.cure, trace='maternalnet', name='fetal')
    newborn = ss.treat(ss.Treatment(efficacy=0.98, transition='cs_cure', name='newborn_tx'), on=syph.born_cs, coverage=0.5, start=2005, name='newborn')
    rpr_test = ss.Test(detects=syph.rpr, sensitivity=0.9, specificity=0.98, name='rpr_test') # Detects titre, not infection. name='rpr' collides with the dimension (see NOTES)
    survey = ss.screen(rpr_test, years=[2025], coverage=0.2, name='rpr_survey')
    return [visits, anc, symp, fetal, newborn, survey]


def make_sim(intvs=True, n_agents=5000, stop=2030, capacity=None, **kwargs):
    syph = Syphilis(name='syph', init_prev=0.02, init=dict(EL=0.01, LL=0.01)) # v3: init_prev (active) and init_latent_prev
    # RPR (non-treponemal) status as a dimension of people: positive on entering secondary (move= above), lost in late latent
    # (v3: lognormal(15, 5) y from late-latent onset; here exponential) or 6-12 months after treatment (v3: early-stage treatment only)
    syph.stratify(rpr=[False, True], split=[1, 0],
                  rpr_wane_late=ss.transfer(rpr=False, dur=ss.years(15), states=['LL', 'T']),
                  rpr_wane_tx=ss.transfer(rpr=False, dur=ss.months(9), states=['S', 'Sp', 'Sr']))
    syph.pars['eff_condom'] = eff_condom # Read by sti.StructuredSexual.net_beta as a plain number (see NOTES)
    sim = ss.Sim(
        syph,
        networks = [sti.StructuredSexual(), ss.MaternalNet()],
        demographics = [ss.Pregnancy(fertility_rate=ss.freqperyear(100)), ss.Deaths(death_rate=ss.freqperyear(10))],
        connectors = ss.modify(syph.rel_trans, multiply=latent_decay, where=syph.EL | syph.LL, name='latent_decay'),
        interventions = make_interventions(syph, capacity) if intvs else None,
        n_agents=n_agents, start=2000, stop=stop, dt=dt, unit='year', verbose=0, **kwargs, # Without unit=, the unit is taken from dt (month) and every bare rate above is read per month (see NOTES)
    )
    return sim


if __name__ == '__main__':
    T = sc.timer()
    base = make_sim(intvs=False).run()
    sim = make_sim().run()
    stockout = make_sim(capacity=ss.peryear(6)).run() # ANC treatment limited to 6 per year: positives beyond it are lost, not queued
    T.toc('Three sims')
    for label, s in [('No interventions', base), ('ANC + treatment', sim)]:
        r = s.results.syph
        infected = r.infected
        print(f'\n{label} (solver={s.solver})')
        print(f'  Infected 2000/2015/2030: {infected[0]:.0f} / {infected[len(infected)//2]:.0f} / {infected[-1]:.0f}')
        print(f'  Stage mix at end: ' + ', '.join(f'{k}={r[f"n_{k}"][-1]:.0f}' for k in ['E', 'Pv', 'Ph', 'S2', 'EL', 'LL', 'T']))
        print(f'  Prevalence at end {r.prev[-1]:.3f}; ever infected {r.ever[-1]:.0f}, of whom cured and trep+ {r.n_Sp[-1]:.0f}, sero-reverted {r.n_Sr[-1]:.0f}')
        print(f'  RPR+ by stage at end: ' + ', '.join(f'{k}={r[f"n_{k}"].by("rpr").values[-1, 1]:.0f}/{r[f"n_{k}"][-1]:.0f}' for k in ['S2', 'EL', 'LL', 'Sp']))
        print(f'  Fetal infections {r.mtct.sum():.0f}: stillbirths {r.stillbirth.sum():.0f}, NND {r.nnd.sum():.0f}, CS births {r.born_cs.sum():.0f}, normal {r.born_ok.sum():.0f}')
        print(f'  Tertiary {r.tertiary.sum():.0f}, deaths from tertiary {r.tert_death.sum():.0f}, reactivations {r.reactivate.sum():.0f}')
        if label != 'No interventions':
            i = s.results
            print(f'  ANC tested {i.anc.delivered.sum():.0f}, positive {i.anc.diagnosed.sum():.0f}, treated {i.anc_positive.delivered.sum():.0f}')
            print(f'  Symptomatic treated {i.symp.delivered.sum():.0f}; cures {r.cure.sum():.0f}; fetal cures {r.fetal_cure.sum():.0f}; newborns cured {r.cs_cure.sum():.0f}')
            print(f'  RPR survey 2025: tested {i.rpr_survey.delivered.sum():.0f}, positive {i.rpr_survey.diagnosed.sum():.0f}')

    i = stockout.results
    print(f'\nStock-out (ANC treatment capacity 6/year): ANC positive {i.anc.diagnosed.sum():.0f}, treated {i.anc_positive.delivered.sum():.0f}; '
          f'CS births {i.syph.born_cs.sum():.0f}, stillbirths {i.syph.stillbirth.sum():.0f}')
