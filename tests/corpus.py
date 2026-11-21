"""
The model corpus: the conformance models of `docs/design/18-conformance.md` in the lingua_franca repo, and a small
model of each standard pattern of the users of other frameworks

Each model is a function that returns an unrun sim, sized to run in seconds; keyword
arguments go to `ss.Sim`, e.g. `corpus.sir(solver='binomial')`. Under `solver='abm'`,
a sim has 5,000 agents unless `n_agents=` is given. `models` lists them all by name.
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

n_agents = 5_000 # The agents for a run as agents, if not given


def make_sim(model, kwargs, **defaults):
    """ A sim of the model, with the defaults overridden by the keyword arguments """
    pars = sc.mergedicts(dict(verbose=0), defaults, kwargs)
    if pars.get('solver') == 'abm':
        pars.setdefault('n_agents', n_agents)
    return ss.Sim(model, **pars)


def make_sir(beta=0.3, dur=10, name='sir', **kwargs):
    """ A simple SIR """
    S, I, R = ss.states('S I R')
    return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, dur=dur), init_prev=0.01, name=name, **kwargs)


#%% The conformance corpus

def sir(**kwargs):
    """ 1. SIR, with no unit anywhere """
    return make_sim(make_sir(), kwargs, stop=100, total_pop=10_000)


def seir(**kwargs):
    """ 2. SEIR, with the source of infection given """
    S, E, I, R = ss.states('S E I R')
    seir = ss.Disease(
        infection = ss.Transition(S, E, rate=0.3 * I),
        latency   = ss.Transition(E, I, dur=5),
        recovery  = ss.Transition(I, R, dur=10),
        init_prev = 0.01,
        name = 'seir',
    )
    return make_sim(seir, kwargs, unit='day', stop=150, total_pop=10_000)


def erlang(**kwargs):
    """ 3. SIR with an Erlang dwell time, exact as stages in every solver """
    sir = make_sir(dur=ss.erlang(mean=10, k=3))
    return make_sim(sir, kwargs, unit='day', stop=100, total_pop=10_000)


def lognormal(**kwargs):
    """ 4. SIR with a lognormal dwell time, run as fitted Erlang stages in a compartmental run """
    sir = make_sir(dur=ss.lognormal(mean=10, std=4, unit='days'))
    return make_sim(sir, kwargs, unit='day', stop=100, total_pop=10_000)


def branching(**kwargs):
    """ 5. A branch on leaving I, with a dwell time in hospital and a competing death hazard """
    S, I, R, H, D = ss.states('S I R H D')
    dis = ss.Disease(
        infection = ss.Transition(S, I, rate=0.3 * I),
        course    = ss.Transition(I, {R: None, H: 0.1}, dur=8),
        discharge = ss.Transition(H, R, dur=10),
        death     = ss.Transition(H, D, rate=ss.perday(0.02)),
        init_prev = 0.01,
        name = 'branching',
    )
    return make_sim(dis, kwargs, unit='day', stop=150, total_pop=10_000)


contact_matrix = np.array([[8, 3, 1], [3, 6, 2], [1, 2, 3]]) # Contacts per day of each age group (rows) with each

def ages(**kwargs):
    """ 6. An age-stratified SIR with a contact matrix """
    sir = make_sir(beta=0.05).stratify(age={0: 0.3, 15: 0.5, 65: 0.2})
    return make_sim(sir, kwargs, unit='day', stop=100, total_pop=100_000, networks=ss.contacts(age=contact_matrix))


def patches(mobility=True, **kwargs):
    """ 7. An SIR on three patches, with gravity mobility or with none """
    towns = ss.geo.geography(names=['north', 'south', 'east'], populations=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_n=dict(I=ss.by(patch=[50, 0, 0])), name='sir')
    sir.stratify(patch=ss.levels(towns, mobility=ss.geo.gravity(scale=2e-4) if mobility else None))
    return make_sim(sir, kwargs, unit='day', stop=150, total_pop=100_000)


def vaccination(**kwargs):
    """ 8. Vaccination as a stratification plus a transfer, which the disease never mentions """
    sir = make_sir().stratify(vaccinated={False: 1, True: 0})
    campaign = ss.Event(dict(vaccinated=True), when=[10], p=0.5, name='campaign')
    return make_sim(sir, kwargs, unit='day', stop=100, total_pop=10_000, interventions=campaign, connectors=ss.modify(sir.infection, multiply=ss.by(vaccinated=[1.0, 0.1])))


def cholera(**kwargs):
    """ 11. Cholera, whose states include a water reservoir, which is a count and not people """
    S, I, R = ss.states(3)
    W = ss.State(people=False) # Bacteria in the water
    cholera = ss.Disease(
        infection = S >> I @ ss.perday(0.5) * W.n / (W.n + 1e5),
        recovery  = I >> R @ ss.days(5),
        shedding  = ss.outside >> W @ 10 * I.n,
        decay     = W >> ss.dead @ ss.days(30),
        init      = dict(I=0.001),
        name = 'cholera',
    )
    return make_sim(cholera, kwargs, unit='day', stop=200, total_pop=10_000)


def make_malaria(mortality=ss.days(10)):
    """ Ross-Macdonald malaria: humans and mosquitoes as states of one disease """
    a, b, c = 0.3, 0.5, 0.5 # Bites per mosquito per day; probabilities that a bite infects a human, and a mosquito
    Sh, Eh, Ih, Rh = ss.states(4)
    Sv, Ev, Iv     = ss.states(3, people=False) # Mosquitoes: counts, not people
    Nh             = (Sh + Eh + Ih + Rh).n
    malaria = ss.Disease(
        infection  = Sh >> Eh @ a * b * Iv.n / Nh,
        latency    = Eh >> Ih @ ss.days(12),
        recovery   = Ih >> Rh @ ss.days(100),
        waning     = Rh >> Sh @ ss.days(365),
        emergence  = ss.unborn >> Sv @ ss.perday(5_000),
        biting     = Sv >> Ev @ a * c * Ih.n / Nh,
        incubation = Ev >> Iv @ ss.days(10),
        mortality  = Sv + Ev + Iv >> ss.dead @ mortality,
        init       = dict(Ih=0.01, Sv=50_000),
        default_state = 'Sh',
        name = 'malaria',
    )
    return malaria


def malaria(**kwargs):
    """ 12. Ross-Macdonald malaria: humans and mosquitoes as states of one disease """
    return make_sim(make_malaria(), kwargs, unit='day', stop=200, total_pop=10_000)


def screening(**kwargs):
    """ 13. Screening in rounds, with treatment of those who test positive """
    S, I = ss.states('S I')
    sis = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, S, dur=10), init_prev=0.01, name='sis')
    test = ss.Diagnostic(sensitivity=0.9, specificity=0.95)
    drug = ss.Treatment(cure=0.95)
    screen = ss.screen(test, when=ss.schedule(every=7, start=30), coverage=0.4, name='screen')
    treatment = ss.treat(drug, when=screen.positive, coverage=0.8, name='treatment')
    return make_sim(sis, kwargs, unit='day', stop=150, total_pop=10_000, interventions=[screen, treatment])


def custom(**kwargs):
    """ 14. A function of the sim in a rate slot, which works on age groups too """
    def severity(sim):
        return 0.1 + 0.3*(sim.people.age > 50)
    S, I, H, R = ss.states('S I H R')
    dis = ss.Disease(
        infection = ss.Transition(S, I, rate=0.3 * I),
        course    = ss.Transition(I, {H: severity, R: None}, dur=7),
        discharge = ss.Transition(H, R, rate=0.1),
        init_prev = 0.01,
        name = 'custom',
    )
    return make_sim(dis, kwargs, unit='day', stop=100, total_pop=10_000, stratify=dict(age=[0, 50]))


def make_outcomes(beta=0.3, likelihood=None):
    beta = ss.par(beta, bounds=[0.05, 1.0])
    S, I, R = ss.states(3)
    infection = S >> I @ beta * I
    recovery  = I >> R @ ss.days(10)
    cases     = ss.outcome(0.5 * infection, delay=ss.days(5), likelihood=likelihood)
    hosp      = ss.outcome(0.05 * infection, delay=ss.days(7), dur=ss.days(10))
    deaths    = ss.outcome(0.2 * hosp, delay=ss.lognormal(mean=14, std=5, unit='days'))
    return ss.Disease(infection, recovery, cases, hosp, deaths, init_prev=0.01, name='sir')

def outcomes(beta=0.3, likelihood=None, data=None, **kwargs):
    """ 15. Outcomes (cases, hospitalizations, deaths), with a parameter that has bounds and, optionally, data """
    return make_sim(make_outcomes(beta, likelihood), kwargs, total_pop=1e6, start='2020-01-01', stop='2020-06-01', data=data)


def make_tb():
    """ TB with drug resistance as a strain, acquired during treatment """
    susceptible, latent, active, on_treatment = ss.states(4)
    tb = ss.Disease(
        infection  = susceptible >> latent @ ss.peryear(8) * active,
        activation = latent >> active @ ss.peryear(0.05),
        diagnosis  = active >> on_treatment @ ss.peryear(0.7),
        cure       = on_treatment >> susceptible @ ss.months(6),
        failure    = on_treatment >> active @ ss.peryear(0.2),
        init       = dict(active=0.01),
        name       = 'tb',
        strains    = ss.Strains(['DS', 'MDR'], name='resistance', fractions=[1.0, 0.0], infection=[1.0, 0.7], cure=[1.0, 0.3]),
    )
    tb.stratify(acquired=ss.transfer(dict(resistance=('DS', 'MDR')), rate=ss.peryear(0.02), states=['on_treatment']))
    return tb


def tb(**kwargs):
    """ 17. TB with drug resistance as a strain, acquired during treatment """
    return make_sim(make_tb(), kwargs, stop=30, total_pop=100_000)


def flu_rsv(**kwargs):
    """ 18. Two diseases in one sim, with age stratified on the sim """
    def make(name, beta, dur):
        S, I = ss.states('S I')
        return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, S, dur=dur), init_prev=0.01, name=name)
    return make_sim([make('flu', 0.4, 5), make('rsv', 0.3, 8)], kwargs, unit='day', stop=100, total_pop=10_000, stratify=dict(age=[0, 5, 18, 65]))


#%% The users' standard patterns

def make_seasonal(b=0.5, dates=None, init=100):
    """ beta as a fitted value, times a seasonal curve, times a time series """
    r = ss.timeseries(dict(zip(dates.strftime('%Y-%m-%d'), np.linspace(1.0, 0.3, len(dates)))))
    b = ss.par(b, bounds=[0.1, 1])
    season = ss.seasonal(mean=0.9, amplitude=0.1/0.9, peak=14/365)
    S, I, R = ss.states('S I R')
    return ss.Disease(infection=ss.Transition(S, I, rate=b*season*r * I), recovery=ss.Transition(I, R, rate=ss.perday(0.25)), init_n=dict(I=init), name='sir')

def seasonal(b=0.5, data=None, **kwargs):
    """ The epydemix and summer2 pattern: beta as a fitted value times `ss.seasonal` times `ss.timeseries` """
    start, stop = '2020-02-23', '2020-05-31'
    dis = make_seasonal(b, dates=pd.date_range(start, stop))
    return make_sim(dis, kwargs, total_pop=1e6, unit='day', start=start, stop=stop, data=data)


def age_vaccine(**kwargs):
    """ The epydemix pattern: an age-stratified model with a vaccine for the under-fives """
    sir = make_sir(beta=0.05).stratify(age={0: 0.15, 5: 0.3, 20: 0.45, 65: 0.1})
    vaccine = ss.vaccinate(ss.Vaccine(efficacy=0.9), who=ss.person.age < 5, when=[10], coverage=0.8, name='vaccinate')
    contacts = np.array([[6, 3, 2, 1], [3, 8, 3, 1], [2, 3, 6, 2], [1, 1, 2, 3]])
    return make_sim(sir, kwargs, unit='day', stop=100, total_pop=100_000, networks=ss.contacts(age=contacts), interventions=vaccine)


def patch_matrix(dim='patch', **kwargs):
    """ The camdl and flepiMoP pattern: two places with an explicit contact matrix over the places, by a dimension called `patch` """
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.1 * I), recovery=ss.Transition(I, R, dur=10), init_n=dict(I=ss.by(**{dim:[10, 0]})), name='sir')
    sir.stratify(**{dim:{'a': 0.5, 'b': 0.5}})
    contacts = ss.contacts(**{dim:[[4.0, 1.0], [1.0, 4.0]]})
    return make_sim(sir, kwargs, unit='day', stop=150, total_pop=20_000, networks=contacts)


def poisson_weekly(**kwargs):
    """ The flepiMoP pattern: a Poisson likelihood on weekly counts that are not whole numbers (here, the expected counts of an ODE run) """
    truth = outcomes().run()
    daily = truth.results.sir.cases.values
    weekly = sc.dataframe(date=truth.t.timevec[0:-6:7], cases=[daily[i:i+7].sum() for i in range(0, len(daily) - 6, 7)]) # The total over the week each data point starts, as a flow is compared
    return outcomes(beta=0.25, likelihood=ss.poisson(), data=weekly, **kwargs)


#%% All of them, by name

models = sc.obj(
    sir=sir, seir=seir, erlang=erlang, lognormal=lognormal, branching=branching, ages=ages, patches=patches, vaccination=vaccination,
    cholera=cholera, malaria=malaria, screening=screening, custom=custom, outcomes=outcomes,
    tb=tb, flu_rsv=flu_rsv, seasonal=seasonal, age_vaccine=age_vaccine, patch_matrix=patch_matrix,
    poisson_weekly=poisson_weekly,
)

# The models that every solver can run, for the tests of agreement between solvers
compartmental = ['sir', 'seir', 'erlang', 'lognormal', 'branching', 'ages', 'patches', 'vaccination', 'cholera', 'malaria', 'screening', 'custom', 'outcomes', 'tb', 'flu_rsv', 'age_vaccine', 'patch_matrix']


#%% The kernels group: the models of test_kernels.py, test_kernel_sets.py, and devtests/test_gradients.py

def make_plain(beta, **kwargs):
    """ An SIR with a dwell time of 8, for the models below """
    S, I, R = ss.states('S I R')
    return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, dur=8), name='sir', **kwargs)


def kernel_sir(**kwargs):
    """ An SIR with reported cases """
    S, I, R = ss.states(3)
    infection = S >> I @ ss.perday(0.4) * I
    recovery  = I >> R @ ss.perday(0.15)
    cases     = ss.outcome(0.6 * infection)
    sir = ss.Disease(infection, recovery, cases, init_n=dict(I=10), name='sir')
    return make_sim(sir, kwargs, total_pop=10_000, unit='day', dur=ss.weeks(12))


def kernel_seir(**kwargs):
    """ Seasonal transmission, a dwell time in stages, and a branch """
    S, E, I, R, D = ss.states('S E I R D')
    seir = ss.Disease(
        infection = ss.Transition(S, E, rate=ss.seasonal(0.4, amplitude=0.3, peak=1) * I),
        latency   = ss.Transition(E, I, dur=ss.erlang(mean=5, k=3)),
        outcome   = ss.Transition(I, {R: None, D: 0.1}, dur=8),
        init      = dict(I=0.01),
        name      = 'seir',
    )
    return make_sim(seir, kwargs, unit='day', stop=200)


def make_towns():
    """ Three towns with commuting between them """
    towns = ss.geo.geography(names=['north', 'south', 'east'], populations=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
    return make_plain(0.3, init_n=dict(I=ss.by(patch=[50, 0, 0]))).stratify(patch=ss.levels(towns, mobility=ss.geo.gravity(scale=2e-4)))


def kernel_towns(**kwargs):
    return make_sim(make_towns(), kwargs, unit='day', stop=150, total_pop=100_000)


def kernel_commute(**kwargs):
    """ Three age groups with a contact matrix within each of three towns, with commuting between them, and aging (which a model without births or deaths has only if asked for) """
    sir = make_towns().stratify(age={0: 0.3, 15: 0.5, 65: 0.2})
    return make_sim(sir, kwargs, unit='day', stop=150, total_pop=100_000, aging=True, networks=ss.contacts(age=contact_matrix))


def kernel_delay(**kwargs):
    """ A fixed latent period, a conveyor, with a death rate acting on it as well """
    S, E, I, R = ss.states('S E I R')
    d = ss.Disease(infection=ss.Transition(S, E, rate=0.4 * I), latency=ss.Transition(E, I, dur=ss.constant(ss.days(5))),
                   recovery=ss.Transition(I, R, dur=ss.days(8)), death=ss.death([E, I], rate=ss.perday(0.02)), init_prev=0.01, name='delay')
    return make_sim(d, kwargs, unit='day', stop=150)


def make_biting(w=(0.5, 2.0)):
    """ A weighted count: each infectious person counted by how much they are bitten (or not weighted, w=None) """
    S, I, R = ss.states('S I R')
    count = I.n if w is None else (ss.by(biting=list(w))*I).n
    sir = ss.Disease(infection=S >> I @ ss.perday(0.3)*count/(S + I + R).n, recovery=I >> R @ ss.days(8), init=dict(I=0.01), name='sir')
    return sir.stratify(biting={'lo': 0.7, 'hi': 0.3})


def kernel_biting(w=(0.5, 2.0), **kwargs):
    return make_sim(make_biting(w), kwargs, unit='day', stop=100, total_pop=10_000)


def kernel_met(**kwargs):
    """ Counts that a contact matrix acts on: the number in each age group (beta*I.n), weighted (by age, and by infectiousness), and that over everyone (beta*I.frac) """
    S, I, R = ss.states('S I R')
    I.rel_trans = 0.5
    sir = ss.Disease(infection=S >> I @ (1e-6*I.n + 1e-6*(ss.by(age=[2, 1, 0.5])*I).n + 2e-6*I.n_infectious + 0.1*I.frac), recovery=I >> R @ ss.days(8), init=dict(I=0.001), name='sir')
    return make_sim(sir.stratify(age={0: 0.3, 15: 0.5, 65: 0.2}), kwargs, unit='day', stop=150, total_pop=100_000, networks=ss.contacts(age=contact_matrix))


kernel_models = sc.obj(sir=kernel_sir, seir=kernel_seir, towns=kernel_towns, commute=kernel_commute, malaria=malaria, tb=tb, # TB has a transfer, so is stepped
                       delay=kernel_delay, biting=kernel_biting, met=kernel_met)


def largest_diff(sim1, sim2):
    """ The largest difference in any result (and its strata) of the declared modules, as a share of that result's largest value """
    out = 0
    for mod in sim1.declared:
        for key,res in mod.results.items():
            if isinstance(res, ss.Result):
                other = sim2.results[mod.name][key]
                for a,b in [(res.values, other.values), (getattr(res, 'strata', None), getattr(other, 'strata', None))]:
                    if a is not None:
                        out = max(out, np.abs(a - b).max()/max(np.abs(b).max(), 1e-12))
    return out


#%% Helpers for the tests of diseases, demographics, dimensions, and the converters (devtests/io)

def make_resistance(beta=4, immunity=True, cure=(0.9, 0.3), **kwargs):
    """ Two strains in one TB-like course: superinfection of the latent, cure for each strain, resistance emerging at failure """
    S, L, I, Tx, R = ss.states('S L I Tx R')
    strains = ss.Strains(['DS', 'RIF'], max_strains=2, fractions=[0.8, 0.2], rel_trans=[1, 0.6], cure=list(cure),
                         immunity=ss.immunity(due_to=L, protection=0.7) if immunity else None)
    DS, RIF = strains
    return ss.Disease(infection=S >> L @ ss.peryear(beta)*I, activation=L >> I @ ss.peryear(0.3), diagnosis=I >> Tx @ ss.peryear(2),
                      treatment=ss.Transition(Tx, {R: 'cure', I: None}, rate=ss.peryear(2), per_strain='each'),
                      failure=ss.Transition(Tx, I, rate=ss.peryear(0.2), move=DS >> (DS, RIF)), relapse=R >> S @ ss.years(2),
                      name='tb', strains=strains, **kwargs)


def io_sir(**kwargs):
    """ An SIR in days, as the converters' tests export it """
    S, I, R = ss.states('S I R')
    dis = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3) * I), recovery=ss.Transition(I, R, dur=ss.days(7)), init_prev=0.01, name='sir')
    return make_sim(dis, kwargs, total_pop=1000, unit='day', stop=60)


def seird(**kwargs):
    """ Births, deaths from every state, a branch, and a lognormal dwell time, which exporters write as Erlang stages """
    S, E, I, R, D = ss.states('S E I R D')
    dis = ss.Disease(
        births    = ss.birth(S, rate=ss.peryear(0.02)),
        infection = ss.Transition(S, E, rate=ss.perday(0.4) * I),
        onset     = ss.Transition(E, I, dur=ss.days(4)),
        outcome   = ss.Transition(I, {R: None, D: 0.1}, dur=ss.lognormal(ss.days(8), ss.days(3))),
        deaths    = ss.death([S, E, I, R], rate=ss.peryear(0.01)),
        init_prev = 0.01, name='seird')
    return make_sim(dis, kwargs, total_pop=1e5, unit='day', stop=120)


#%% Helpers for the tests of the sim, runs, results, and attributes

class BySex(ss.Analyzer):
    """ A 2D result: the number infected with the SIS by sex, at the start of each step """
    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('infected', columns=['female', 'male'], label='Infected', kind='stock'))

    def record_states(self):
        infected = self.sim.diseases.sis.I
        female = self.sim.people.female
        self.results.infected[self.ti] = [np.count_nonzero(infected & female), np.count_nonzero(infected & ~female)]


def make_hiv(init_prev=0.05, init=None, art=ss.peryear(0.2)):
    """ HIV with CD4 as attributes (with the parameters of the STIsim port): drawn at infection, recorded when starting and stopping ART, and derived by state """
    cd4_max, cd4_healthy, art_cd4_growth = 1000, 500, 0.1 # art_cd4_growth is per month
    eff_art, time_to_eff, rel_trans_aids, rel_art_mort = 0.99, 0.5, 8.0, 0.25
    class HIV(ss.Disease):
        S        = ss.State()
        acute    = ss.State(rel_trans=5.3)
        latent   = ss.State()
        falling  = ss.State()
        on_art   = ss.State()
        post_art = ss.State()
        infected = acute + latent + falling + on_art + post_art # A sum of states, as a source and for states=
        infection   = S >> acute @ ss.peryear(0.15)*(acute + latent + falling + on_art + post_art)
        latency     = ss.Transition(acute, latent, dur=ss.lognormal(ss.months(1.7), ss.months(1)), stages=3)
        progression = ss.Transition(latent, falling, dur=ss.lognormal(ss.years(10), ss.years(3)))
        aids_death  = ss.Transition(falling, ss.dead, dur=ss.lognormal(ss.years(3), ss.years(1)), stages=9)
        start_art   = (acute + latent + falling + post_art) >> on_art @ art
        dropout     = ss.Transition(on_art, post_art, dur=ss.lognormal(ss.years(3), ss.years(1.5)))

        cd4_start     = ss.attr(ss.normal(800, 50), when=infection)
        cd4_latent    = ss.attr(ss.normal(500, 50), when=infection)
        cd4_preart    = ss.attr('cd4', when=start_art) # CD4 just before starting
        cd4_potential = ss.attr(cd4_preart*(1 + (cd4_max - cd4_preart)/cd4_healthy*np.log(cd4_max/cd4_preart)), when=start_art, combine='first') # Set at the first start only
        cd4_stop      = ss.attr('cd4', when=dropout)
        gain = cd4_potential - cd4_preart
        cd4 = ss.attr({
            acute:    cd4_start - (cd4_start - cd4_latent)*acute.progress,
            latent:   cd4_latent,
            falling:  np.maximum(1, cd4_latent*(1 - falling.progress)),
            on_art:   cd4_preart + gain*(2/(1 + np.exp(-art_cd4_growth*on_art.time_in/ss.months(1))) - 1),
            post_art: np.maximum(1, cd4_stop*(1 - post_art.progress)),
        })
        post_art_death = ss.Transition(post_art, ss.dead, dur=np.maximum(np.log(cd4_preart)*cd4_stop/cd4_potential, 0.1)*ss.years(1), stages=10)
        cd4_death      = infected >> ss.dead @ ss.by(cd4=ss.peryear([0.30, 0.05, 0.01, 0.005, 0.003]))

    hiv = HIV(init_prev=init_prev, init=init, name='hiv')
    hiv.stratify(cd4=ss.levels([0, 50, 200, 350, 500], states=hiv.infected)) # Derived: no groups
    mods = [
        ss.modify(hiv.cd4_death, multiply=rel_art_mort, who=hiv.on_art, name='art_mort'),
        ss.modify(hiv.rel_trans, multiply=1 - eff_art*np.minimum(hiv.on_art.time_in/ss.years(time_to_eff), 1), who=hiv.on_art, name='art_trans'),
        ss.modify(hiv.rel_trans, multiply=rel_trans_aids, who=(hiv.cd4 < 200) & (hiv.falling | hiv.post_art), name='aids_trans'),
    ]
    return hiv, mods


#%% The interventions group: helpers for test_interventions.py and test_deliveries.py (and their devtests)

def make_sis(beta=0.3, dur=10, name='sis', **kwargs):
    """ A simple SIS """
    S, I = ss.states('S I')
    return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, S, dur=dur), init_prev=0.01, name=name, **kwargs)


def make_geno(name, beta=1.2):
    """ An HPV genotype as a disease of its own, S -> I -> S """
    S, I = ss.states('S I')
    return ss.Disease(infection=ss.Transition(S, I, rate=ss.peryear(beta) * I), clearance=I >> S @ ss.years(1), init_prev=0.1, name=name)


def make_hpv(name, beta=1.5):
    """ An HPV genotype with a lesion and cancer: S -> I -> CIN -> CA, cleared to S, and excised (undriven) back to I """
    S, I, CIN, CA = ss.states('S I CIN CA')
    return ss.Disease(
        infection = S >> I @ ss.peryear(beta) * (I + CIN),
        clearance = I >> S @ ss.years(1),
        dysplasia = I >> CIN @ ss.years(2),
        regress   = CIN >> S @ ss.years(3),
        invasion  = CIN >> CA @ ss.years(5),
        excised   = CIN >> I @ 0,
        init_prev=0.1, name=name,
    )


def run_each(make, solvers=('ode', 'binomial', 'abm'), **kwargs):
    """ Run the sim that `make` returns with each solver """
    return sc.obj({solver:make(solver=solver, **kwargs).run() for solver in solvers})


def check_close(sims, func, rtol=0.1):
    """ Check that a number from each sim (by solver) matches the ODE's """
    vals = sc.obj({solver:func(sim) for solver,sim in sims.items()})
    assert np.allclose(vals[:], vals.ode, rtol=rtol), f'Expected the solvers to agree, not {vals}'
    return vals


def check_agents(make, **kwargs):
    """ Check that a model is agents only: an explicit ODE refuses it before the run, and solver='auto' runs it as agents """
    import pytest
    with pytest.raises(ss.CapabilityError):
        make(solver='ode', **kwargs).init()
    assert isinstance(make(**kwargs).init().solver, ss.solvers.abm)


# Events and clocks (test_events.py and devtests/test_events.py)
class BV(ss.Disease):
    """ BV, whose recurrence after treatment decays with the time since it, P.time_in """
    CST1, CST4, P = ss.states('CST1 CST4 P')
    default_state = 'CST1'
    recur0, recur_tau = ss.par(ss.permonth(0.06)), ss.par(ss.months(3))
    onset  = CST1 >> CST4 @ ss.permonth(0.05)
    tx     = CST4 >> P @ ss.months(2) # Treated after two months, on average: a dwell time, which ends in the state-change row
    recur  = ss.Transition(P, CST4, rate=recur0*np.exp(-P.time_in/recur_tau)) # Largest on entry
    settle = ss.Transition(P, CST1, dur=ss.months(6))


#%% Helpers for the tests of dimensions

def at_start(sim):
    """ Initialize a sim, and return its results with the number in each state at the start, at index 0, before the first step """
    sim.init()
    for mod in sim.declared:
        mod.engine.start_step()
        mod.engine.record_states()
    return sim.results


def run_with(mod, solver, **kwargs):
    """ Run a module with one solver, as 10,000 agents if it is agent-based """
    pars = dict(n_agents=10_000) if solver == 'abm' else {}
    return ss.Sim(mod, solver=solver, **pars, verbose=0, **kwargs).run()


def stages(res, state):
    """ A state's counts, summed over its stages (I_1, I_2, ...) where an exporter wrote it as them (for devtests/io) """
    return res[state].values if state in res else sum(res[k].values for k in res.keys() if k.startswith(f'{state}_') and k[len(state)+1:].isdigit())
