"""
The model corpus: the conformance models of `design/18-conformance.md`, and a small
model of each standard pattern of the users in the switching review (`v4_design/oct17/switching/`)

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
    sir = make_sir(dur=ss.lognorm_ex(mean=10, std=4, unit='days'))
    return make_sim(sir, kwargs, unit='day', stop=100, total_pop=10_000)


def branching(**kwargs):
    """ 5. A branch on leaving I, with a dwell time in hospital and a competing death hazard """
    S, I, R, H, D = ss.states('S I R H D')
    dis = ss.Disease(
        infection = ss.Transition(S, I, rate=0.3 * I),
        course    = ss.Transition(I, [R, H], dur=8, p=[None, 0.1]),
        discharge = ss.Transition(H, R, dur=10),
        death     = ss.Transition(H, D, rate=ss.perday(0.02)),
        init_prev = 0.01,
        name = 'branching',
    )
    return make_sim(dis, kwargs, unit='day', stop=150, total_pop=10_000)


contact_matrix = np.array([[8, 3, 1], [3, 6, 2], [1, 2, 3]]) # Contacts per day of each age group (rows) with each

def ages(**kwargs):
    """ 6. An age-stratified SIR with a contact matrix """
    sir = make_sir(beta=0.05).stratify(age=[0, 15, 65], split=[0.3, 0.5, 0.2])
    return make_sim(sir, kwargs, unit='day', stop=100, total_pop=100_000, networks=ss.contacts(age=contact_matrix))


def patches(mobility=True, **kwargs):
    """ 7. An SIR on three patches, with gravity mobility or with none """
    towns = ss.geo.geography(names=['north', 'south', 'east'], pop=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_n=dict(I=ss.by(patch=[50, 0, 0])), name='sir')
    sir.stratify(patch=towns, mobility=ss.geo.gravity(k=2e-4) if mobility else None)
    return make_sim(sir, kwargs, unit='day', stop=150, total_pop=100_000)


def vaccination(**kwargs):
    """ 8. Vaccination as a stratification plus a transfer, which the disease never mentions """
    sir = make_sir().stratify(vaccinated=[False, True], split=[1, 0], adjust={'infection': [1.0, 0.1]})
    campaign = ss.Delivery(move=dict(vaccinated=True), coverage=0.5, at=[10], name='campaign')
    return make_sim(sir, kwargs, unit='day', stop=100, total_pop=10_000, interventions=campaign)


def coinfection(**kwargs):
    """ 9. HIV and syphilis, each modifying the other, over a sexual network """
    S, I = ss.states('S I')
    hiv = ss.Disease(infection=ss.Transition(S, I, rate=dict(sexual=ss.peract(m2f=0.002, f2m=0.001)) * I), init_prev=0.05, name='hiv')
    S, I = ss.states('S I')
    syph = ss.Disease(infection=ss.Transition(S, I, rate=0.01 * I), cure=ss.Transition(I, S, dur=ss.years(3)), init_prev=0.05, name='syph')
    connectors = [
        ss.modify(hiv.rel_trans, multiply=20.0, where=syph.I, name='hiv_trans'),
        ss.modify(syph.rel_sus, multiply=5.0, where=hiv.I, name='syph_sus'),
    ]
    return make_sim([hiv, syph], kwargs, networks=ss.SexualNet(), connectors=connectors, n_agents=2_000, start=2000, stop=2010)


def network(**kwargs):
    """ 10. An SIS over a sexual network, which 'auto' runs as agents """
    S, I = ss.states('S I')
    sis = ss.Disease(infection=ss.Transition(S, I, rate=dict(sexual=ss.peract(m2f=0.08, f2m=0.05)) * I), recovery=ss.Transition(I, S, dur=ss.years(5)), init_prev=0.05, name='sis')
    return make_sim(sis, kwargs, networks=ss.SexualNet(), n_agents=2_000, start=2000, stop=2010)


def cholera(**kwargs):
    """ 11. Cholera, whose states include a water reservoir, which is a count and not people """
    S, I, R = ss.states(3)
    W = ss.State(people=False) # Bacteria in the water
    cholera = ss.Disease(
        infection = S >> I @ ss.perday(0.5) * W.n / (W.n + 1e5),
        recovery  = I >> R @ ss.days(5),
        shedding  = ss.null >> W @ 10 * I.n,
        decay     = W >> ss.null @ ss.days(30),
        init      = dict(I=0.001),
        name = 'cholera',
    )
    return make_sim(cholera, kwargs, unit='day', stop=200, total_pop=10_000)


def malaria(**kwargs):
    """ 12. Ross-Macdonald malaria: humans and mosquitoes as states of one disease """
    a, b, c = 0.3, 0.5, 0.5 # Bites per mosquito per day; probabilities that a bite infects a human, and a mosquito
    Sh, Eh, Ih, Rh = ss.states(4)
    Sv, Ev, Iv     = ss.states(3, people=False) # Mosquitoes: counts, not people
    Nh             = (Sh + Eh + Ih + Rh).n
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
        first = 'Sh',
        name = 'malaria',
    )
    return make_sim(malaria, kwargs, unit='day', stop=200, total_pop=10_000)


def screening(**kwargs):
    """ 13. Screening in rounds, with treatment of those who test positive """
    S, I = ss.states('S I')
    sis = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, S, dur=10), init_prev=0.01, name='sis')
    test = ss.Diagnostic(sensitivity=0.9, specificity=0.95)
    drug = ss.Treatment(cure=0.95)
    screen = ss.screen(test, coverage=0.4, interval=7, start=30, name='screen')
    treatment = ss.treat(drug, who=screen.positive, coverage=0.8, name='treatment')
    return make_sim(sis, kwargs, unit='day', stop=150, total_pop=10_000, interventions=[screen, treatment])


def custom(**kwargs):
    """ 14. A function of the sim in a rate slot, which works on age groups too """
    def severity(sim):
        return 0.1 + 0.3*(sim.people.age > 50)
    S, I, H, R = ss.states('S I H R')
    dis = ss.Disease(
        infection = ss.Transition(S, I, rate=0.3 * I),
        course    = ss.Transition(I, [H, R], p=severity, dur=7),
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
    hosp      = ss.outcome(0.05 * infection, delay=ss.days(7), duration=ss.days(10))
    deaths    = ss.outcome(0.2 * hosp, delay=ss.lognorm_ex(mean=14, std=5, unit='days'))
    return ss.Disease(infection, recovery, cases, hosp, deaths, init_prev=0.01, name='sir')

def outcomes(beta=0.3, likelihood=None, data=None, **kwargs):
    """ 15. Outcomes (cases, hospitalizations, deaths), with a parameter that has bounds and, optionally, data """
    return make_sim(make_outcomes(beta, likelihood), kwargs, total_pop=1e6, start='2020-01-01', stop='2020-06-01', data=data)


def threshold(**kwargs):
    """ 16. Distancing while more than 1,000 are infected, as a function of the sim """
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.001, name='sir')
    distancing = ss.modify(sir.pars.beta, multiply=0.3, when=lambda sim: sim.results.sir.I[sim.ti - 1] > 1000, name='distancing')
    return make_sim(sir, kwargs, unit='day', stop=200, total_pop=50_000, interventions=distancing)


def tb(**kwargs):
    """ 17. TB with drug resistance as a strain, acquired during treatment """
    susceptible, latent, active, on_treatment = ss.states(4)
    tb = ss.Disease(
        infection  = susceptible >> latent @ ss.peryear(8) * active,
        activation = latent >> active @ ss.peryear(0.05),
        diagnosis  = active >> on_treatment @ ss.peryear(0.7),
        cure       = on_treatment >> susceptible @ ss.months(6),
        failure    = on_treatment >> active @ ss.peryear(0.2),
        init       = dict(active=0.01),
        name       = 'tb',
        strains    = ss.Strains(['DS', 'MDR'], name='resistance', split=[1.0, 0.0], infection=[1.0, 0.7], cure=[1.0, 0.3]),
    )
    tb.stratify(acquired=ss.transfer(resistance=('DS', 'MDR'), rate=ss.peryear(0.02), states=['on_treatment']))
    return make_sim(tb, kwargs, stop=30, total_pop=100_000)


def flu_rsv(**kwargs):
    """ 18. Two diseases in one sim, with age stratified on the sim """
    def make(name, beta, dur):
        S, I = ss.states('S I')
        return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, S, dur=dur), init_prev=0.01, name=name)
    return make_sim([make('flu', 0.4, 5), make('rsv', 0.3, 8)], kwargs, unit='day', stop=100, total_pop=10_000, stratify=dict(age=[0, 5, 18, 65]))


#%% The users' standard patterns

def make_seasonal(b=0.5, dates=None):
    """ beta as a fitted value, times a seasonal curve, times a time series """
    r = ss.timeseries(dict(zip(dates.strftime('%Y-%m-%d'), np.linspace(1.0, 0.3, len(dates)))))
    b = ss.par(b, bounds=[0.1, 1])
    season = ss.seasonal(mean=0.9, amplitude=0.1/0.9, peak=14/365)
    S, I, R = ss.states('S I R')
    return ss.Disease(infection=ss.Transition(S, I, rate=b*season*r * I), recovery=ss.Transition(I, R, rate=ss.perday(0.25)), init_n=dict(I=100), name='sir')

def seasonal(b=0.5, data=None, **kwargs):
    """ The epydemix and summer2 pattern: beta as a fitted value times `ss.seasonal` times `ss.timeseries` """
    start, stop = '2020-02-23', '2020-05-31'
    dis = make_seasonal(b, dates=pd.date_range(start, stop))
    return make_sim(dis, kwargs, total_pop=1e6, unit='day', start=start, stop=stop, data=data)


def age_vaccine(**kwargs):
    """ The epydemix pattern: an age-stratified model with a vaccine for the under-fives """
    sir = make_sir(beta=0.05).stratify(age=[0, 5, 20, 65], split=[0.15, 0.3, 0.45, 0.1])
    vaccine = ss.vaccinate(ss.Vaccine(efficacy=0.9), where=ss.People(1).age < 5, coverage=0.8, at=[10], name='vaccinate')
    contacts = np.array([[6, 3, 2, 1], [3, 8, 3, 1], [2, 3, 6, 2], [1, 1, 2, 3]])
    return make_sim(sir, kwargs, unit='day', stop=100, total_pop=100_000, networks=ss.contacts(age=contacts), interventions=vaccine)


def patch_matrix(dim='patch', **kwargs):
    """ The camdl and flepiMoP pattern: two places with an explicit contact matrix over the places, by a dimension called `patch` """
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.1 * I), recovery=ss.Transition(I, R, dur=10), init_n=dict(I=ss.by(**{dim:[10, 0]})), name='sir')
    sir.stratify(**{dim:['a', 'b']}, split=[0.5, 0.5])
    contacts = ss.contacts(**{dim:np.array([[4.0, 1.0], [1.0, 4.0]])})
    return make_sim(sir, kwargs, unit='day', stop=150, total_pop=20_000, networks=contacts)


def poisson_weekly(**kwargs):
    """ The flepiMoP pattern: a Poisson likelihood on weekly counts that are not whole numbers (here, the expected counts of an ODE run) """
    truth = outcomes().run()
    daily = truth.results.sir.cases.values
    weekly = sc.dataframe(date=truth.t.timevec[7::7], cases=[daily[i-6:i+1].sum() for i in range(7, len(daily), 7)]) # The total since the last data point, as a flow is compared
    return outcomes(beta=0.25, likelihood=ss.poisson(), data=weekly, **kwargs)


def starsim3(**kwargs):
    """ The Starsim 3 pattern: an SIR written as code over a random network """
    return make_sim(None, kwargs, diseases=ss.SIR(beta=ss.perday(0.1), init_prev=0.01), networks=ss.RandomNet(n_contacts=4), n_agents=5_000, start=2000, stop=2002)


#%% All of them, by name

models = sc.objdict(
    sir=sir, seir=seir, erlang=erlang, lognormal=lognormal, branching=branching, ages=ages, patches=patches, vaccination=vaccination,
    coinfection=coinfection, network=network, cholera=cholera, malaria=malaria, screening=screening, custom=custom, outcomes=outcomes,
    threshold=threshold, tb=tb, flu_rsv=flu_rsv, seasonal=seasonal, age_vaccine=age_vaccine, patch_matrix=patch_matrix,
    poisson_weekly=poisson_weekly, starsim3=starsim3,
)

# The models that every solver can run, for the tests of agreement between solvers
compartmental = ['sir', 'seir', 'erlang', 'lognormal', 'branching', 'ages', 'patches', 'vaccination', 'cholera', 'malaria', 'screening', 'custom', 'outcomes', 'tb', 'flu_rsv', 'age_vaccine', 'patch_matrix']
