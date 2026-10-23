"""
Benchmark models for the performance guard (test_baselines.py) and the S0.3 benchmarks (v4_design/oct20/simplify/bench.py)

Each covers a kind of model the other benchmarks miss:

- fseis10: malariasim's "fseis10" scenario, ported to Starsim alone (see make_fseis())
- counts:  a declared model with counts (Ross-Macdonald malaria, mosquitoes as counts), as agents
- coded:   a disease written as code (an ss.Infection subclass), over a network, with demographics
- by_age:  a declared model with ss.by age tables (Covasim's prognoses, FPsim's deaths by age and sex)
- hpv:     genotypes as ss.Strains, with attributes and branch chances that depend on a clock (HPVsim)
- tb:      a declared disease as agents with four transmissions over one network, at 1,000 and 10,000 agents (tbsim)
- contra:  a declared choice among many states, by regression and tables by age and destination (FPsim's contraception)
- strains_ode: 16 sets of strains under the ODE (tbsim's drug resistance)
"""
import numpy as np
import starsim as ss
import starsim.library as ssl


#%% fseis10: malariasim's SEIS humans with ODE mosquitoes

class MosquitoODE(ss.Module):
    """ malariasim's MosquitoesODE for one place: SEI mosquitoes stepped by Euler, with a monthly emergence; gives the humans' EIR """
    def __init__(self, emergence, a=0.3, life=10.0, eip=12.0):
        super().__init__()
        self.emergence, self.a, self.mu, self.sigma = np.asarray(emergence), a, 1/life, 1/eip # Per human per day; bites per day; per day
        self.p_inf = 0.0 # The chance a bite infects a mosquito, written by the humans
        self.eir = 0.0 # Infectious bites per person per step, read by the humans
        return

    def init_post(self):
        super().init_post()
        self.S, self.E, self.I = self.rate_now()*len(self.sim.people)/self.mu, 0.0, 0.0 # At the disease-free equilibrium
        return

    def rate_now(self):
        """ The emergence rate now, interpolated between months """
        m = (self.t.now('year') % 1)*12 - 0.5
        i = int(np.floor(m)) % 12
        return self.emergence[i] + (m - np.floor(m))*(self.emergence[(i + 1) % 12] - self.emergence[i])

    def step(self):
        dt = self.t.dt.days
        n = self.sim.people.alive.count()
        lam = self.a*self.p_inf
        S, E, I = self.S, self.E, self.I
        self.S = max(S + (self.rate_now()*n - lam*S - self.mu*S)*dt, 0)
        self.E = max(E + (lam*S - self.sigma*E - self.mu*E)*dt, 0)
        self.I = max(I + (self.sigma*E - self.mu*I)*dt, 0)
        self.eir = self.a*self.I*dt/max(n, 1)
        return


class HumanSEIS(ss.Infection):
    """ malariasim's MalariaHumanSEIS: bitten from the mosquitoes' EIR, with biting heterogeneity by person and age, written as code """
    def __init__(self, **kwargs):
        super().__init__()
        self.define_pars(
            b = 0.5, # The chance an infectious bite infects a human
            c = 0.5, # The chance a bite on an infectious human infects a mosquito
            dur_exp = ss.normal(loc=ss.days(12), scale=ss.days(2)),
            dur_inf = ss.lognorm_ex(mean=ss.days(180), std=ss.days(60)),
            p_bite = ss.bernoulli(p=0), # Set each step
            het = ss.gamma(a=2, scale=0.5), # Biting heterogeneity, mean 1
        )
        self.update_pars(**kwargs)
        self.define_states(
            ss.BoolState('exposed'),
            ss.FloatArr('ti_infectious'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('heterogeneity', default=0.0),
            ss.FloatArr('rel_bite', default=1.0),
            ss.FloatArr('infectiousness', default=0.0),
        )
        return

    def step_state(self):
        alive = self.sim.people.alive.uids
        new = alive[self.heterogeneity[alive] == 0] # The initial people, and those born since the last step
        if len(new):
            self.heterogeneity[new] = np.maximum(self.pars.het.rvs(new), 1e-10)
        age = self.sim.people.age[alive]
        raw = self.heterogeneity[alive]*np.interp(age, [0, 5, 20], [0.3, 0.6, 1.0])
        self.rel_bite[alive] = raw/raw.mean()
        infectious = (self.exposed & (self.ti_infectious <= self.ti)).uids
        self.exposed[infectious] = False
        self.infected[infectious] = True
        recovered = (self.infected & (self.ti_recovered <= self.ti)).uids
        self.infected[recovered] = False
        self.susceptible[recovered] = True
        self.infectiousness[self.infected] = 1.0
        self.infectiousness[self.susceptible] = 0.0
        return

    def step(self):
        mos = self.sim.connectors.mosquitoode
        uids = self.susceptible.uids
        self.pars.p_bite.set(p=np.clip(mos.eir*self.pars.b*self.rel_bite[uids], 0, 1))
        self.set_prognoses(self.pars.p_bite.filter(uids))
        alive = self.sim.people.alive.uids
        mos.p_inf = self.pars.c*np.mean(self.rel_bite[alive]*self.infectiousness[alive])
        return

    def set_prognoses(self, uids, sources=None):
        super().set_prognoses(uids, sources)
        self.exposed[uids] = True
        self.ti_infected[uids] = self.ti
        self.ti_infectious[uids] = self.ti + self.pars.dur_exp.rvs(uids)
        self.ti_recovered[uids] = self.ti_infectious[uids] + self.pars.dur_inf.rvs(uids)
        return


def make_fseis(n_agents=10_000, years=2):
    """
    malariasim's "fseis10": SEIS humans as agents, written as code, and ODE mosquitoes, one place, daily steps, 2 years, crn=False

    malariasim itself is 6,000 lines, so this keeps what costs time per step (per-agent arrays written from
    Python: biting heterogeneity by age, infectiousness, the bite draw) and leaves out what does not (its
    campaign classes, symptoms and node bookkeeping). The outbreak at day 30 is an initial prevalence instead.
    """
    eir = 6.0/365 # A target EIR of 6 a year
    emergence = eir/(0.3*0.5*0.4)/10*np.array([0.5, 0.5, 0.7, 1.0, 1.4, 1.8, 1.8, 1.4, 1.0, 0.7, 0.5, 0.5]) # Per human per day, seasonal
    return ss.Sim(diseases=HumanSEIS(init_prev=0.15), connectors=MosquitoODE(emergence), demographics=True, n_agents=n_agents,
                  start=2000, stop=2000 + years, dt=ss.days(1), use_aging=False, rand_seed=0, verbose=0)


#%% counts: a declared model with counts

def make_counts(n_agents=10_000):
    """ Ross-Macdonald malaria with mosquitoes as counts and humans as agents (malaria_recheck.md MA39), a year in days """
    a, b, c = 0.3, 0.5, 0.5
    Sh, Eh, Ih, Rh = ss.states(4)
    Sv, Ev, Iv = ss.states(3, people=False)
    Nh = (Sh + Eh + Ih + Rh).n
    malaria = ss.Disease(
        infection  = Sh >> Eh @ a * b * Iv.n / Nh,
        latency    = Eh >> Ih @ ss.days(12),
        recovery   = Ih >> Rh @ ss.days(100),
        waning     = Rh >> Sh @ ss.days(365),
        emergence  = ss.null >> Sv @ ss.perday(5_000),
        biting     = Sv >> Ev @ a * c * Ih.n / Nh,
        incubation = Ev >> Iv @ ss.days(10),
        mortality  = Sv + Ev + Iv >> ss.null @ ss.days(10),
        init       = dict(Ih=0.01, Sv=50_000),
        first      = 'Sh',
        name = 'malaria',
    )
    return ss.Sim(malaria, solver='abm', unit='day', start=0, stop=365, n_agents=n_agents, rand_seed=1, verbose=0)


#%% coded: a disease written as code

def make_coded(n_agents=20_000):
    """ A disease written as code (ssl.AgentSIS: per-agent waning immunity) over a random network, with births and deaths, 20 years monthly """
    return ss.Sim(diseases=ssl.AgentSIS(beta=ss.peryear(0.1), init_prev=0.05), networks='random', demographics=True,
                  n_agents=n_agents, start=2000, stop=2020, dt=ss.months(1), rand_seed=1, verbose=0)


#%% by_age: ss.by age tables

cutoffs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
prognoses = dict( # Covasim's defaults, rounded
    sus   = [0.34, 0.67, 1, 1, 1, 1, 1.24, 1.47, 1.47, 1.47],
    symp  = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.9],
    sev   = [0.001, 0.003, 0.012, 0.032, 0.049, 0.102, 0.166, 0.243, 0.273, 0.273],
    crit  = [0.06, 0.06, 0.05, 0.05, 0.063, 0.122, 0.274, 0.432, 0.709, 0.709],
    death = [0.667, 0.25, 0.276, 0.267, 0.286, 0.38, 0.401, 0.425, 0.465, 0.465],
)

def make_by_age(n_agents=20_000):
    """ Covasim's natural history with its age tables of prognoses and susceptibility (CV11), and deaths by single year of age and sex times a trend (FP57), 120 days """
    by_age = lambda key: ss.by(dict(age=cutoffs, value=prognoses[key]))
    ages = np.arange(100)
    mx = 0.0005*np.exp(0.07*ages)
    mortality = ss.by(dict(age=np.r_[ages, ages], sex=['female']*100 + ['male']*100, value=ss.peryear(np.r_[mx, 1.2*mx]))) * ss.timeseries({2020: 1.0, 2021: 0.9})
    dur = lambda days: ss.lognorm_ex(mean=ss.days(days), std=ss.days(days/3))
    S, E, Ia, Ip, Im, Is, Ic, R = ss.states('S E Ia Ip Im Is Ic R')
    covid = ss.Disease(
        infection = S >> E @ ss.probperday(0.016) * (Ia + Ip + Im + Is + Ic),
        to_asymp  = ss.Transition(E, Ia, dur=dur(4.5), p=1 - by_age('symp')),
        to_presym = ss.Transition(E, Ip, dur=dur(4.5), p=by_age('symp')),
        asym_rec  = ss.Transition(Ia, R, dur=dur(8)),
        to_mild   = ss.Transition(Ip, Im, dur=dur(1.1), p=1 - by_age('sev')),
        to_sev    = ss.Transition(Ip, Is, dur=dur(6.6), p=by_age('sev')),
        mild_rec  = ss.Transition(Im, R, dur=dur(8)),
        sev_rec   = ss.Transition(Is, R, dur=dur(18), p=1 - by_age('crit')),
        to_crit   = ss.Transition(Is, Ic, dur=dur(1.5), p=by_age('crit')),
        crit_rec  = ss.Transition(Ic, R, dur=dur(18), p=1 - by_age('death')),
        death     = ss.Transition(Ic, ss.null, dur=dur(10), p=by_age('death')),
        init = dict(E=0.001), name='covid',
    )
    sus = ss.modify(covid.rel_sus, multiply=by_age('sus'), name='sus_by_age')
    return ss.Sim(diseases=covid, connectors=sus, demographics=[ss.death(rate=mortality)], networks=ss.RandomNet(n_contacts=20), n_agents=n_agents,
                  start='2020-03-01', dur=ss.days(120), dt=ss.days(1), use_aging=False, rand_seed=0, verbose=0)


#%% hpv: genotypes as ss.Strains, with attributes and branch chances of a clock

def make_hpv(n_agents=10_000, n_genotypes=4):
    """
    HPVsim's natural history as declared (the 2026-10-22 review's hpv_v4.py, condensed): genotypes as own courses of one
    disease, sharing cancer, with cross-immunity from attributes set at clearance, and branch chances that depend on the
    time spent in a state (HP63), 2000-2040 quarterly
    """
    yrs = ss.years(1)
    lognorm = lambda mean, std: ss.lognorm_ex(mean=ss.years(mean), std=ss.years(std))
    def logf2(x, k, ttc=50): # Severity after x years (hpvsim.utils.logf2)
        t1 = 1 + np.exp(-k*ttc)
        lo, hi = t1/(t1 - 2), -t1/(t1 - 2)
        return np.minimum(1, lo + (hi - lo)/(1 + np.exp(-k*x)))
    def genotype(name, rel_beta, k, tp):
        S, P, C, Ca, L, X = ss.states('S P C Ca L X')
        sev = np.abs(ss.ppl.rel_sev)
        p_cin = logf2(sev*P.time/yrs, k)
        p_ca = 1 - (1 - tp)**(logf2(sev*C.time/yrs, k)**2)
        precin = ss.Transition(P, [C, L, S], dur=lognorm(2.5, 9), p=[p_cin, 0*p_cin, None], where=ss.ppl.female)
        cin = ss.Transition(C, [Ca, L, S], dur=lognorm(5, 20), p=[p_ca, 0*p_ca, None])
        cleared = precin.to(S) | cin.to(S)
        sero = ss.attr(ss.bernoulli(0.6), on=cleared, combine='max', init=0)
        return ss.Disease(S, P, C, Ca, L, X, name=name,
            infection    = ss.Transition(S, P, rate={'randomnet': ss.peryear(0.4*rel_beta)}*(P + C)),
            male_clear   = ss.Transition(P, S, dur=lognorm(1, 1), where=~ss.ppl.female),
            precin       = precin,
            cin          = cin,
            reactivation = ss.Transition(L, P, rate=ss.probperyear(0.025)),
            cancer_death = ss.Transition(Ca, ss.null, dur=lognorm(8, 3)),
            sero         = sero,
            nab          = ss.attr(ss.uniform(0.5, 0.95)*sero, on=cleared, combine='max', init=0),
            cell         = ss.attr(ss.uniform(0.3, 0.9), on=cleared, combine='max', init=0),
            infected     = P + C,
        )
    specs = [('hpv16', 1.0, 0.3, 2e-3), ('hpv18', 0.75, 0.25, 2e-3), ('hi5', 0.9, 0.2, 1.5e-3), ('ohr', 0.9, 0.2, 1.5e-3)][:n_genotypes]
    members = [genotype(*spec) for spec in specs]
    n = len(members)
    sus = np.full((n, n), 0.3) + 0.6*np.eye(n)
    sev = np.full((n, n), 0.5) + 0.4*np.eye(n)
    immunity = [ss.immunity(due_to='nab', rel_sus=sus, combine='sum'), ss.immunity(due_to='cell', dur=dict(P=sev), combine=lambda p: np.minimum(p.sum(0), 0.999))]
    hpv = ss.Disease(name='hpv', init_prev=0.1, strains=ss.Strains(members, shared=['Ca'], immunity=immunity, name='genotype'))
    people = ss.People(n_agents, age_data=ss.uniform(0, 60), extra_states=[ss.FloatArr('rel_sev', default=ss.normal(1.0, 0.2))])
    return ss.Sim(diseases=hpv, people=people, networks=ss.RandomNet(n_contacts=2), start=2000, stop=2040, dt=ss.years(0.25), rand_seed=1, verbose=0)


#%% tb: tbsim's LSHTM spectrum of disease, as agents on a network

def make_tb(n_agents=10_000):
    """
    tbsim's declared TB (Schwalb et al. 2025) as agents: four transmissions (into infection from susceptible, cleared,
    recovered and treated), each over the same random network, with births and deaths as agents, 1990-2020 at
    dt = 0.1 years, as tbsim's own benchmark (tests/profile_sim.py; TS4, TS5)
    """
    sus, inf, cle, rec, non, asy, sym, trt, trd = ss.states('susceptible infection cleared recovered non_infectious asymptomatic symptomatic treatment treated')
    foi = ss.peryear(2.4)*(0.82*asy + sym)
    tb = ss.Disease(
        transmission = sus >> inf @ foi,
        reinfect_cle = cle >> inf @ 1.0*foi,
        reinfect_rec = rec >> inf @ 0.21*foi,
        reinfect_trd = trd >> inf @ 3.15*foi,
        clearance    = inf >> cle @ ss.peryear(1.90),
        latent_non   = inf >> non @ ss.peryear(0.16),
        latent_asy   = inf >> asy @ ss.peryear(0.06),
        recovery     = non >> rec @ ss.peryear(0.18),
        worsening    = non >> asy @ ss.peryear(0.25),
        improvement  = asy >> non @ ss.peryear(1.66),
        symptoms     = asy >> sym @ ss.peryear(0.88),
        remission    = sym >> asy @ ss.peryear(0.54),
        tb_death     = ss.Transition(sym, ss.null, rate=ss.peryear(0.34)),
        diagnosis    = ss.Transition([non, asy, sym], trt, rate=ss.peryear(0.1)),
        completion   = trt >> trd @ ss.peryear(2.0),
        init = dict(infection=0.05, asymptomatic=0.005, symptomatic=0.005), name='tb',
    )
    return ss.Sim(diseases=tb, demographics=[ss.Births(), ss.Deaths()], networks=ss.RandomNet(n_contacts=10), solver='abm',
                  n_agents=n_agents, start=1990, stop=2020, dt=ss.years(0.1), rand_seed=1, verbose=0)


#%% contra: FPsim's contraceptive choice, declared

def make_contra(n_agents=10_000):
    """
    FPsim's contraceptive choice as declared (the 2026-10-22 review's contra_v4b.py, with made-up coefficients): one state
    per method, durations of use by age, and at each exit a choice whose chances are a logistic regression on traits and
    age times a switching table by age and destination, 2000-2010 monthly (FP82)
    """
    methods = ['pill', 'iud', 'inj', 'cond', 'implant', 'other']
    st = {n: ss.State(n) for n in ['young', 'male', 'never', 'none'] + methods}
    dests = [st['none']] + [st[m] for m in methods]
    never_kind = [st['never']] + dests[1:]
    ppl = ss.ppl
    ages = np.arange(15, 50)
    age_bins = [0, 18, 20, 25, 35]
    def p_use(ever):
        age = ss.by(dict(age=np.r_[0, ages], value=np.r_[-1.0, 0.8*np.sin(ages/10)]))
        rhs = -1.2 + 0.9*ever + 0.3*ppl.urban + 0.1*np.minimum(ppl.parity, 5) + age
        return 1/(1 + np.exp(-rhs))
    rng = np.random.default_rng(0)
    w = ss.by(to=dict(zip(methods, 0.5 + rng.random(len(methods)))))
    def mix(src):
        vals = rng.random(len(age_bins)*len(methods)) + 0.01
        return ss.by(dict(age=np.repeat(age_bins, len(methods)), to=methods*len(age_bins), value=vals))
    def branch(src, ever):
        pu = p_use(ever)
        return [1 - pu, pu*mix(src)*w]
    def dur(i):
        scale = ss.by(dict(age=age_bins, value=np.exp(2.0 + 0.2*i + 0.1*np.arange(len(age_bins)))))
        return ss.weibull(c=1.5, scale=scale, unit='month') if i % 2 else ss.gamma(a=2.0, scale=scale/2, unit='month')
    fecund = ppl.female & (ppl.age >= 15) & (ppl.age <= 50)
    tr = dict(
        sexed      = ss.Transition(st['young'], st['male'], dur=0, where=~ppl.female),
        debut      = ss.Transition(st['young'], never_kind, dur=0, p=branch('none', 0), where=fecund & (ppl.age >= 18)),
        from_never = ss.Transition(st['never'], never_kind, dur=dur(0), p=branch('none', 0), where=fecund),
        from_none  = ss.Transition(st['none'], dests, dur=dur(0), p=branch('none', 1), where=fecund),
    )
    for i, m in enumerate(methods):
        tr[f'from_{m}'] = ss.Transition(st[m], dests, dur=dur(i + 1), p=branch(m, 1), where=fecund)
    women = sum([s for k, s in st.items() if k not in ['male', 'young']], st['young'])
    users = sum([st[m] for m in methods[1:]], st[methods[0]])
    contra = ss.Disease(*st.values(), **tr, cpr=users/women, first='young', name='contra')
    people = ss.People(n_agents, extra_states=[ss.BoolState('urban', default=ss.bernoulli(0.3)), ss.FloatArr('parity', default=ss.poisson(2))])
    return ss.Sim(diseases=contra, people=people, start=2000, stop=2010, dt=ss.months(1), use_aging=True, rand_seed=1, verbose=0)


#%% strains_ode: a TB-like course with four strains carried in pairs, as an ODE

def make_strains_ode(substeps=1):
    """
    Drug resistance as in tbsim with two drugs: four strains (drug-susceptible, resistant to either, and to both), up to
    two carried at once, so 16 sets of strains, with superinfection, cure by strain and resistance emerging at failure,
    under the ODE, 2000-2020 in steps of 0.1 years (TR4); substeps=1 since the automatic substeps take minutes here (plan_b C2)
    """
    S, L, I, Tx, R = ss.states('S L I Tx R')
    strains = ss.Strains(['DS', 'RIF', 'INH', 'MDR'], max_strains=2, split=[0.85, 0.05, 0.05, 0.05], rel_trans=[1, 0.6, 0.8, 0.5], cure=[0.9, 0.4, 0.7, 0.3],
                         immunity=ss.immunity(due_to=L, rel_sus=0.7))
    DS, RIF, INH, MDR = strains
    tb = ss.Disease(infection=S >> L @ ss.peryear(4)*I, activation=L >> I @ ss.peryear(0.3), diagnosis=I >> Tx @ ss.peryear(2),
                    treatment=ss.Transition(Tx, [R, I], p='cure', rate=ss.peryear(2), per='strain'),
                    fail_rif=ss.Transition(Tx, I, rate=ss.peryear(0.1), move=DS >> (DS, RIF)),
                    fail_inh=ss.Transition(Tx, I, rate=ss.peryear(0.1), move=DS >> (DS, INH)),
                    relapse=R >> S @ ss.years(2), init=dict(I=0.02), name='tb', strains=strains)
    return ss.Sim(diseases=tb, solver='ode', substeps=substeps, total_pop=1e5, start=2000, stop=2020, dt=ss.years(0.1), verbose=0)
