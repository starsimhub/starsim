"""
STIsim/HIVsim HIV, ported to the Starsim v4 API (stress test, not production).

Covers: stisim/diseases/hiv.py (natural history, CD4, ART effects, mortality),
stisim/interventions/hiv_interventions.py (HIVTest, ART, VMMC, Prep), the hivsim
defaults (MF + maternal + breastfeeding networks, pregnancy + deaths), and the
Zimbabwe setup (ART numbers on treatment, VMMC numbers, testing ramps).

Run: conda run -n starsim python port.py
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

datadir = sc.path('/Users/robynstuart/gf/stisim/hivsim_examples/zimbabwe')
T0 = sc.tic()

# %% Parameters (from stisim HIVPars; per year unless typed)
cd4_death_bins  = np.array([1000, 500, 350, 200, 50, 0])          # descending, as v3 np.digitize uses them
cd4_death_rates = np.array([0.003, 0.003, 0.005, 0.01, 0.05, 0.30])
rel_death_f     = 0.74
rel_art_mort    = 0.25    # effective ART keeps 25% of off-ART CD4 mortality
art_cd4_growth  = 0.1     # per month (v3 uses timesteps; v3 default dt is a month)
cd4_max, cd4_healthy = 1000, 500
eff_art         = 0.99    # transmission efficacy of suppressive ART
time_to_eff     = 0.5     # years to full ART efficacy (linear ramp)
rel_trans_aids  = 8.0     # v3 applies this to cd4 < 200, not to the falling stage


def cd4_hazard(sim, uids):
    """ CD4-dependent HIV mortality (per year), off and on ART: v3 make_p_hiv_death + get_art_mortality_hazard """
    hiv = sim.diseases.hiv
    rate = cd4_death_rates[np.digitize(hiv.cd4[uids], cd4_death_bins)]
    rate = rate*np.where(sim.people.female[uids], rel_death_f, 1.0)
    # ART is a state here, so its mortality benefit is multiplied in by hand (as a product, efficacy=dict(mortality=0.75) does it)
    rate = rate*np.where(hiv.on_art[uids], rel_art_mort, 1.0)
    return rate


def post_art_dur(sim, uids):
    """ Mean time from stopping ART to AIDS death, from CD4 when stopping (v3 HIV.stop_art) """
    hiv = sim.diseases.hiv
    cd4 = np.maximum(hiv.cd4[uids], 1)
    pre = np.maximum(hiv.cd4_preart[uids], 2)
    pot = np.maximum(hiv.cd4_potential[uids], 1)
    return np.maximum(np.log(pre)*cd4/pot, 0.1)   # years


# %% The disease: declared stages + transitions, with CD4 as custom per-agent state
class HIV(ss.Disease):
    S        = ss.State()
    acute    = ss.State(rel_trans=5.3)
    latent   = ss.State()
    falling  = ss.State()
    on_art   = ss.State()                 # rel_trans set per agent (efficacy ramp), see art_rel_trans
    post_art = ss.State()                 # v3 'art_discontinued': off ART after having been on it

    beta = dict(mf=[0.035, 0.0175], maternal=0.3, breastfeeding=0.06)   # per year; MF pair is [m2f, f2m]
    infection   = S >> acute @ beta * (acute + latent + falling + on_art + post_art)
    latency     = acute >> latent @ ss.lognorm_ex(ss.months(1.7), ss.months(1))
    progression = latent >> falling @ ss.lognorm_ex(ss.years(10), ss.years(3))
    aids_death  = falling >> ss.null @ ss.lognorm_ex(ss.years(3), ss.years(1))       # v3 ti_zero
    start_art   = (acute | latent | falling | post_art) >> on_art @ 0                 # only ever forced by ART delivery
    dropout     = on_art >> post_art @ ss.lognorm_ex(ss.years(3), ss.years(1.5))      # v3 dur_on_art
    post_art_death = ss.Transition(post_art, ss.null, dur=post_art_dur, stages=100)   # Erlang(100) ~ v3 normal(mean, 0.1*mean)
    cd4_death   = (acute | latent | falling | on_art | post_art) >> ss.null @ cd4_hazard

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.define_states(
            ss.FloatArr('cd4'),
            ss.FloatArr('cd4_start', default=ss.normal(800, 50)),
            ss.FloatArr('cd4_latent'),
            ss.FloatArr('cd4_preart'),
            ss.FloatArr('cd4_potential'),
            ss.FloatArr('cd4_postart'),
            ss.FloatArr('care_seeking', default=ss.normal(1, 0.5)),
            ss.BoolState('diagnosed'),
            ss.FloatArr('ti_diagnosed'),
            ss.BoolState('ever_art'),
        )
        self.cd4_latent_dist = ss.normal(500, 50)
        self.cd4_falling_dist = ss.uniform(0.05, 1)
        return

    @property
    def infected(self):
        return self.acute | self.latent | self.falling | self.on_art | self.post_art

    def init_post(self):
        super().init_post()
        self.care_seeking[:] = np.maximum(self.care_seeking[:], 0.01)
        # Seeded people start mid-course (v3 back-dates infections 5 mo to 10 y): give them a CD4 to match their stage
        for state in [self.acute, self.latent, self.falling]:
            uids = state.uids
            self.cd4_latent[uids] = self.cd4_latent_dist.rvs(uids)
            self.cd4[uids] = self.cd4_latent[uids]
        f = self.falling.uids
        self.cd4[f] = self.cd4_latent[f]*self.cd4_falling_dist.rvs(f)
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('deaths_cd4_lt200', dtype=float, label='HIV deaths, CD4 < 200', scale=True),
            ss.Result('deaths_cd4_ge200', dtype=float, label='HIV deaths, CD4 >= 200', scale=True),
            ss.Result('new_diagnoses', dtype=float, label='New diagnoses', scale=True),
            ss.Result('p_on_art', dtype=float, label='Proportion of PLHIV on ART', scale=False),
            ss.Result('mean_cd4_art', dtype=float, label='Mean CD4 on ART', scale=False),
        )
        return

    def step_state(self):
        """ Declared transitions (super), wrapped by the per-agent CD4 trajectory (v3 HIV.step_state) """
        ti, dt = self.ti, self.t.dt.years
        ev = self.engine.events

        # Entry hook for infection: must run BEFORE super(), since super() evaluates cd4_death's hazard (which reads cd4)
        new_inf = ev.get('infection', ss.uids())        # infected during last step's transmission; popped in step()
        if len(new_inf):
            self.cd4[new_inf] = self.cd4_start[new_inf]
            self.cd4_latent[new_inf] = self.cd4_latent_dist.rvs(new_inf)

        super().step_state()

        # BUG WORKAROUND: agents added by ss.Pregnancy (newborns, and unborn agents at conception) are put in NO state
        # of a declared disease -- only ss.birth() places people. Put them in the first state by hand.
        orphans = (self.sim.people.alive & ~self.S & ~self.infected).uids
        if len(orphans):
            self.S[orphans] = True
            self.ti_S[orphans] = ti

        # Entry hooks: whoever entered a state since last step
        new_art = ev.get('start_art', ss.uids())        # forced by the ART delivery last step
        if len(new_art):
            self.cd4_preart[new_art] = np.maximum(self.cd4[new_art], 1)
            naive = new_art[~self.ever_art[new_art]]
            pre = self.cd4_preart[naive]
            self.cd4_potential[naive] = pre + pre*(cd4_max - pre)/cd4_healthy*np.log(cd4_max/pre)
            self.ever_art[new_art] = True
        stopped = ev.get('dropout', ss.uids())
        if len(stopped):
            self.cd4_postart[stopped] = self.cd4[stopped]

        # Trajectories, closed-form from entry/exit times. ti_<state> and ti_exit_<state> come free from the engine
        def frac(state, uids):  # share of the dwell in this state that has elapsed
            t0, t1 = self[f'ti_{state}'][uids], self[f'ti_exit_{state}'][uids]
            return np.clip((ti - t0)/np.maximum(t1 - t0, 1), 0, 1)
        a = self.acute.uids
        a = a[~np.isnan(self.ti_exit_acute[a])]  # GOTCHA: ti_exit_<state> is drawn at the start of the NEXT step, so is NaN on the step of entry
        self.cd4[a] = self.cd4_start[a] - (self.cd4_start[a] - self.cd4_latent[a])*frac('acute', a)
        l = self.latent.uids
        self.cd4[l] = self.cd4_latent[l]
        f = self.falling.uids
        f = f[~np.isnan(self.ti_exit_falling[f])]
        self.cd4[f] = np.maximum(1, self.cd4_latent[f]*(1 - frac('falling', f)))
        o = self.on_art.uids
        if len(o):
            t_art = (ti - self.ti_on_art[o])     # steps; v3 growth rate is per step
            gain = self.cd4_potential[o] - self.cd4_preart[o]
            self.cd4[o] = 2*gain/(1 + np.exp(-t_art*art_cd4_growth)) - gain + self.cd4_preart[o]
        p = self.post_art.uids
        p = p[~np.isnan(self.ti_exit_post_art[p])]
        self.cd4[p] = np.maximum(1, self.cd4_postart[p]*(1 - frac('post_art', p)))
        return

    def update_results(self):
        super().update_results()
        ti = self.ti
        res = self.results
        res.new_diagnoses[ti] = np.count_nonzero(self.ti_diagnosed == ti)   # scale=True: scaled to people at the end
        n_inf = self.infected.count()
        res.p_on_art[ti] = self.on_art.count()/n_inf if n_inf else 0
        o = self.on_art.uids
        res.mean_cd4_art[ti] = np.nanmean(self.cd4[o]) if len(o) else np.nan
        return


# Deaths by CD4: no per-agent record of who left by which transition after they're removed, so count at the event
class DeathsByCD4(ss.Analyzer):
    def step(self):
        hiv = self.sim.diseases.hiv
        ev = hiv.engine.events
        uids = ss.uids.concatenate([ev.get(k, ss.uids()) for k in ['cd4_death', 'aids_death', 'post_art_death']])
        if len(uids):
            lo = np.count_nonzero(hiv.cd4.raw[uids] < 200)
            hiv.results.deaths_cd4_lt200[self.ti] += lo
            hiv.results.deaths_cd4_ge200[self.ti] += len(uids) - lo
        return


# %% ART effect on transmission: a modifier with a per-agent amount (ramp over 6 months), instead of v3 update_transmission()
def art_rel_trans(sim):
    hiv = sim.diseases.hiv
    t_on = (sim.ti - hiv.ti_on_art.raw)*sim.t.dt.years
    eff = eff_art*np.clip(t_on/time_to_eff, 0, 1)
    return np.nan_to_num(1 - eff, nan=1.0)

def aids_rel_trans(sim):
    hiv = sim.diseases.hiv
    return np.where(hiv.cd4.raw < 200, rel_trans_aids, 1.0)


def make_modifiers(hiv):
    return [
        ss.modify(hiv.rel_trans, multiply=art_rel_trans, where=hiv.on_art, name='art_trans'),
        ss.modify(hiv.rel_trans, multiply=aids_rel_trans, where=hiv.falling | hiv.post_art, name='aids_trans'),
    ]


# %% Testing (HIVTest): time-varying annual test probabilities, as per-step coverage
def annual_to_step(years, probs):
    """ ss.screen coverage is per round, so an annual probability has to be converted by hand """
    def cov(sim):
        p = np.interp(sim.t.now('year'), years, probs)
        return 1 - (1 - p)**sim.t.dt.years
    return cov

def diagnose(sim, uids):
    hiv = sim.diseases.hiv
    hiv.diagnosed[uids] = True
    hiv.ti_diagnosed[uids] = sim.ti
    return


def make_testing():
    yrs = np.array([1990, 2020, 2040])
    hiv_test = ss.Test(sensitivity=1.0, specificity=1.0)
    gp_test = ss.screen(hiv_test, interval=ss.months(1), coverage=annual_to_step(yrs, [0, 0.5, 0.6]),
                        eligible=lambda sim: ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art,
                        on_positive=diagnose, name='gp_test')
    cd4_test = ss.screen(hiv_test, interval=ss.months(1), coverage=annual_to_step(yrs, [0, 0.85, 0.95]),
                         eligible=lambda sim: (sim.diseases.hiv.cd4 < 200) & ~sim.diseases.hiv.diagnosed,
                         on_positive=diagnose, name='cd4_test')
    return [gp_test, cd4_test]


# %% ART (sti.ART): hit a STOCK target (number on ART, from data), prioritizing low CD4 / high care-seeking
art_data = pd.read_csv(datadir/'art_coverage.csv').dropna()

def art_target(sim):
    """ Number on ART this step, in agents """
    n = np.interp(sim.t.now('year'), art_data.year, art_data.n_art)
    return n/sim.current_scale

def art_deliver(sim):
    hiv = sim.diseases.hiv
    n_on = hiv.on_art.count()
    gap = art_target(sim) - n_on
    if gap < 0: # Over target: v3 removes the highest-CD4 / lowest-care-seeking. No delivery can do this; force the dropout transition by hand
        on = hiv.on_art.uids
        drop = on[np.argsort(-hiv.cd4[on]/hiv.care_seeking[on])[:int(-gap)]]
        hiv.engine.force(drop, hiv.graph.transitions['dropout'], hiv.graph.states['post_art'])
        return ss.uids()
    cands = (hiv.diagnosed & ~hiv.on_art).uids
    cands = cands[hiv.infected[cands]]
    order = np.argsort(hiv.cd4[cands]/hiv.care_seeking[cands], kind='stable')
    return cands[order[:int(round(gap))]]


def make_art():
    art = ss.Treatment(efficacy=1.0, transition='start_art')   # "cures" = forces start_art; re-initiation from post_art for free
    return ss.treat(art, deliver=art_deliver, name='art_prog')  # name='art' collides with the product's name (named from its variable)


# %% VMMC: the Zimbabwe data are annual numbers of circumcisions (a FLOW), so capacity= fits them directly
def make_vmmc():
    vmmc = ss.Vaccine(efficacy=0.6, name='vmmc')                # rel_sus x 0.4 for those who hold it
    vmmc_data = pd.read_csv(datadir/'vmmc_coverage.csv')
    cap = ss.timeseries(dict(zip(vmmc_data.year, vmmc_data.n_vmmc)))  # people per year
    men = lambda sim: sim.people.male & (sim.people.age >= 15) & (sim.people.age < 50)
    return ss.vaccinate(vmmc, eligible=men, capacity=cap, start=2008, name='vmmc_prog')


# %% PrEP: product with efficacy, held for a course; AGYW instead of FSW (no FSW in MFNet)
def make_prep():
    prep = ss.Vaccine(efficacy=0.8, waning=ss.months(3), name='prep')   # exponential loss, not a fixed 3-month course
    agyw = lambda sim: sim.people.female & (sim.people.age >= 15) & (sim.people.age < 25) & ~sim.diseases.hiv.infected
    return ss.vaccinate(prep, eligible=agyw, coverage=annual_to_step([2015, 2020, 2025], [0, 0.1, 0.3]), interval=ss.months(1), start=2015, name='prep_prog')


# %% Sim
def make_sim(n_agents=5000, stop=2020, maternal=True, **kwargs):
    hiv = HIV(init=dict(acute=0.002, latent=0.04, falling=0.01), name='hiv')
    ppl = ss.People(n_agents, age_data=pd.read_csv(datadir/'zimbabwe_age_1990.csv'))
    asfr = pd.read_csv(datadir/'zimbabwe_asfr.csv')
    deaths = pd.read_csv(datadir/'zimbabwe_deaths.csv')
    demog = [ss.Pregnancy(fertility_rate=asfr, metadata=dict(data_cols=dict(year='Time', age='AgeStart', value='Value'))),
             ss.Deaths(death_rate=deaths, rate_units=1, metadata=dict(data_cols=dict(year='Time', sex='Sex', age='AgeStart', value='Value')))]
    agyw_sus = ss.modify(hiv.rel_sus, multiply=1.7, where=ppl.female & (ppl.age >= 15) & (ppl.age < 25), name='agyw_sus')  # v3 rel_sus_age
    nets = [ss.MFNet(duration=ss.lognorm_ex(ss.years(2), ss.years(2)))]  # default 15-year partnerships saturate
    if maternal:
        nets += [ss.MaternalNet(), ss.BreastfeedingNet()]
    intvs = make_testing() + [make_art(), make_vmmc(), make_prep()]
    sim = ss.Sim(
        diseases=hiv, people=ppl, networks=nets, demographics=demog,
        connectors=make_modifiers(hiv) + [agyw_sus], interventions=intvs, analyzers=DeathsByCD4(),
        start=1990, stop=stop, dt=ss.months(1), unit='year', total_pop=10.1e6, verbose=0, **kwargs,
    )
    return sim


# %% Alternatives, to answer the review questions (small runs)

def alt_capacity_as_flow():
    """ Q1: n_art is a STOCK. capacity= is a FLOW (people per unit time); feeding it the stock overshoots, and picks at random """
    art = ss.Treatment(efficacy=1.0, transition='start_art')
    cap = ss.timeseries(dict(zip(art_data.year, art_data.n_art)), mode='linear')
    elig = lambda sim: sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art
    sim = make_sim(n_agents=2000, stop=2012, maternal=False)
    sim.pars.interventions = make_testing() + [ss.treat(art, eligible=elig, capacity=cap, name='art_prog')]
    sim.run()
    r = sim.results.hiv
    i = sc.findnearest(sim.t.yearvec, 2011.9)
    print(f'  capacity=n_art as a flow: n_on_art 2012 = {r.n_on_art[i]:,.0f} vs stock target {np.interp(2012, art_data.year, art_data.n_art):,.0f}; '
          f'starts {sim.results.art_prog.delivered.sum():,.0f}; eligible-but-untreated at end {sim.results.art_prog.eligible[i]:,.0f}')
    return sim


def alt_priority_lost():
    """ Q1: deliver= returns candidates sorted by CD4; capacity= then caps them AT RANDOM, so the priority order is lost """
    starts = {}
    for label in ['sorted in deliver=, capped inside', 'sorted in deliver=, capped by capacity=']:
        art = ss.Treatment(efficacy=1.0, transition='start_art')
        def all_sorted(sim):
            hiv = sim.diseases.hiv
            c = (hiv.diagnosed & ~hiv.on_art).uids
            c = c[hiv.infected[c]]
            return c[np.argsort(hiv.cd4[c], kind='stable')]
        k = 100_000  # people per year
        if label.endswith('inside'):
            def deliver(sim):
                n = int(k*sim.t.dt.years/sim.current_scale + 0.5)
                return all_sorted(sim)[:n]
            prog = ss.treat(art, deliver=deliver, name='art_prog')
        else:
            prog = ss.treat(art, deliver=all_sorted, capacity=ss.peryear(k), name='art_prog')
        sim = make_sim(n_agents=2000, stop=2008, maternal=False)
        sim.pars.interventions = make_testing() + [prog]
        sim.run()
        hiv = sim.diseases.hiv
        ever = hiv.ever_art.uids
        starts[label] = np.nanmean(hiv.cd4_preart[ever]) if len(ever) else np.nan
        print(f'  {label}: mean CD4 at ART start = {starts[label]:.0f} ({len(ever)} agents)')
    return starts


class HIVNoART(ss.Disease):
    """ Natural history only: ART is a held product, not a state """
    S       = ss.State()
    acute   = ss.State(rel_trans=5.3)
    latent  = ss.State()
    falling = ss.State(rel_trans=8)
    infection   = S >> acute @ dict(mf=[0.035, 0.0175]) * (acute + latent + falling)
    latency     = acute >> latent @ ss.lognorm_ex(ss.months(1.7), ss.months(1))
    progression = latent >> falling @ ss.lognorm_ex(ss.years(10), ss.years(3))
    aids_death  = falling >> ss.null @ ss.lognorm_ex(ss.years(3), ss.years(1))
    other_death = (acute | latent | falling) >> ss.null @ ss.peryear(0.005)


def alt_art_as_product():
    """ Q3/Q4: ART as ss.Treatment(efficacy=dict(...)): transmission + mortality efficacy, exponential dropout via waning=, re-initiation """
    hiv = HIVNoART(init=dict(latent=0.05, falling=0.01), name='hiv')
    artp = ss.Treatment(efficacy=dict(transmission=0.96, mortality=0.75), waning=ss.years(3), name='artp')
    on = ss.Cond('dim', 'has_artp')  # no public handle for "holds the product": spell the dim name
    pause = [ss.modify(hiv.progression, multiply=0, where=on, name='pause_prog'),   # ART halts progression (clock paused, not reset)
             ss.modify(hiv.latency, multiply=0, where=on, name='pause_lat')]
    def deliver(sim):
        held = sim.people['has_artp']
        n_on = np.count_nonzero(held.raw[sim.people.auids])
        gap = int(round(np.interp(sim.t.now('year'), art_data.year, art_data.n_art)/sim.current_scale - n_on))
        h = sim.diseases.hiv
        c = (h.latent | h.falling | h.acute).uids
        c = c[held[c] == 0]
        return c[:max(gap, 0)]
    prog = ss.treat(artp, deliver=deliver, name='art_prog')
    sim = ss.Sim(hiv, networks=ss.MFNet(duration=ss.lognorm_ex(ss.years(2), ss.years(2))), interventions=prog, connectors=pause,
                 n_agents=2000, start=1990, stop=2015, dt=ss.months(1), unit='year', total_pop=10.1e6, verbose=0)
    sim.run()
    held = sim.people['has_artp']
    starts = sim.results.art_prog.delivered.sum()
    print(f'  product ART: on ART at end {np.count_nonzero(held.raw[sim.people.auids])*sim.current_scale:,.0f}, starts {starts:,.0f} '
          f'(> stock, so re-initiation after waning works); mortality efficacy hits: {[m.name for m in sim.modifiers[("hiv", "transition", "aids_death")]]}')
    return sim


def alt_diagnosed_dimension():
    """ Diagnosis as a declared dimension + ss.transfer on_positive, instead of a BoolState written by a function """
    S, acute, latent = ss.states(3)
    hiv = ss.Disease(infection=S >> acute @ dict(mf=[0.035, 0.0175]) * (acute + latent),
                     latency=acute >> latent @ ss.months(2), init=dict(latent=0.05), name='hiv')
    hiv.stratify(dx=[False, True], states=[acute, latent])
    test = ss.Test(sensitivity=1.0)
    screening = ss.screen(test, interval=ss.months(1), coverage=0.05, eligible=~hiv.dx, on_positive=ss.transfer(dx=True, coverage=1.0), name="screening")
    try:
        sim = ss.Sim(hiv, networks=ss.MFNet(), interventions=screening, n_agents=1000, start=2000, stop=2005, unit='year', dt=ss.months(1), verbose=0).run()
        r = sim.results.hiv
        print(f'  dx as a dimension: diagnosed {sim.results.screening.diagnosed.sum():,.0f}; n_latent by dx at end {r.n_latent.by("dx").values[-1]}')
    except Exception as E:
        print(f'  dx as a dimension: FAILS {type(E).__name__}: {str(E)[:300]}')
    return


def alt_coverage_as_annual_prob():
    """ Can ss.screen take an annual probability (ss.probperyear) as coverage, so the per-step conversion is automatic? """
    S, I = ss.states(2)
    d = ss.Disease(infection=S >> I @ 0.5*I, init=dict(I=0.2), name='d')
    try:
        scr = ss.screen(ss.Test(), interval=ss.months(1), coverage=ss.probperyear(0.5), name='scr')
        sim = ss.Sim(d, interventions=scr, n_agents=2000, start=2000, stop=2002, dt=ss.months(1), unit='year', verbose=0).run()
        print(f'  coverage=ss.probperyear(0.5), monthly rounds: mean per-round coverage {np.mean(sim.results.scr.coverage[1:-1]):.3f} (0.056 if converted, 0.5 if not)')
    except Exception as E:
        print(f'  coverage=ss.probperyear: FAILS {type(E).__name__}: {str(E)[:300]}')


def alt_beta_dt():
    """ Bug check: is per-act transmission on MFNet independent of dt? """
    out = {}
    for dt in [ss.weeks(1), ss.months(1), ss.years(0.25)]:
        S, I = ss.states(2)
        d = ss.Disease(infection=S >> I @ dict(mf=[0.05, 0.05]) * I, init=dict(I=0.1), name='d')
        sim = ss.Sim(d, networks=ss.MFNet(duration=ss.lognorm_ex(ss.years(2), ss.years(2))), n_agents=3000, start=2000, stop=2005,
                     dt=dt, unit='year', verbose=0).run()
        out[str(dt)] = sim.results.d.infection.sum()
    print(f'  infections in 5 y by dt (should be ~equal): {out}')
    return out


if __name__ == '__main__':
    try:
        sim = make_sim(maternal=True)
        sim.run()
    except Exception as E:
        print(f'With maternal + breastfeeding networks: {type(E).__name__}: {E}')
        sim = make_sim(maternal=False)
        sim.run()

    print(f'solver: {sim.solver} ({sim.solver_reason}); {sc.toc(T0, output=True):.1f} s')
    r = sim.results.hiv
    yrs = sim.t.yearvec
    idx = [sc.findnearest(yrs, y) for y in [1990, 1995, 2000, 2005, 2010, 2015, 2019.9]]
    df = sc.dataframe(
        year=yrs[idx].round(1),
        prev=[(r.prevalence[i]).round(3) for i in idx],
        n_on_art=r.n_on_art[idx].round(),
        art_target=[np.interp(yrs[i], art_data.year, art_data.n_art).round() for i in idx],
        p_on_art=r.p_on_art[idx].round(3),
        cd4_art=r.mean_cd4_art[idx].round(),
    )
    print(df)
    ann = lambda x: x.sum()  # whole run
    print('infections:', ann(r.infection).round(), ' dx:', (sim.results.gp_test.diagnosed.sum() + sim.results.cd4_test.diagnosed.sum()).round())
    print('deaths  aids:', ann(r.aids_death).round(), ' cd4:', ann(r.cd4_death).round(), ' post-ART:', ann(r.post_art_death).round())
    print('deaths by CD4 <200 / >=200:', ann(r.deaths_cd4_lt200).round(), ann(r.deaths_cd4_ge200).round())
    hiv = sim.diseases.hiv
    kids = (sim.people.age < 3) & hiv.infected
    print('infected children <3 at end (vertical transmission):', kids.count())
    print('ART starts:', sim.results.art_prog.delivered.sum().round(), ' dropouts:', ann(r.dropout).round(),
          ' VMMC:', sim.results.vmmc_prog.delivered.sum().round(), ' PrEP starts:', sim.results.prep_prog.delivered.sum().round())

    print('\nAlternatives:')
    for fn in [alt_beta_dt, alt_capacity_as_flow, alt_priority_lost, alt_art_as_product, alt_diagnosed_dimension, alt_coverage_as_annual_prob]:
        try:
            fn()
        except Exception as E:
            print(f'  {fn.__name__}: FAILS {type(E).__name__}: {str(E)[:400]}')
    print(f'total {sc.toc(T0, output=True):.1f} s')
