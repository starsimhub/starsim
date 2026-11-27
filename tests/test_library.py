"""
Test the Starsim library (starsim.library, i.e. ssl): each module runs, and its headline mechanism does something
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import pytest
import starsim as ss
import starsim.library as ssl

n_agents = 2_000

endemic_pars  = dict(start=2000, stop=2030, n_agents=n_agents, rand_seed=1, verbose=0)


@sc.timer()
def test_hiv():
    """ Check that HIV runs, and that ART reduces deaths """
    sc.heading('Testing HIV, ART, and CD4_analyzer...')

    def make(label, interventions=None):
        return ss.Sim(label=label, interventions=interventions, **endemic_pars,
                      diseases = ssl.HIV(beta=ss.peryear(0.05), init_prev=0.05),
                      networks = ss.RandomNet(n_contacts=ss.poisson(2)),
                      analyzers = ssl.CD4_analyzer())

    s1 = make('No ART')
    s2 = make('With ART', ssl.ART(coverage=ss.timeseries({2005: 0, 2015: 0.9}, 'linear')))
    for sim in [s1, s2]:
        sim.run()

    assert s2.results.art.n_art.max() > 0, 'Expected some agents on ART'
    assert s2.results.hiv.death.sum() < s1.results.hiv.death.sum(), 'Expected ART to reduce HIV deaths'

    # The analyzer should have recorded plausible CD4 counts
    cd4 = s1.analyzers.cd4_analyzer.cd4
    assert cd4.shape[0] == s1.t.npts, 'Expected one CD4 row per timestep'
    assert 0 < cd4.mean() <= 500, f'Expected CD4 counts in (0, 500], not {cd4.mean()}'

    return s1, s2


@sc.timer()
def test_networks():
    """ Library networks, as a sim would use them """
    sc.heading('Testing library networks')
    n = 400
    nets = dict(geo=ssl.GeoNet('Kenya', n_contacts=6), grid=ssl.GridNet(periodic=True), small=ssl.SmallWorldNet(n_contacts=4), free=ssl.ScaleFreeNet(n_contacts=4))
    sims = {k:ss.Sim(n_agents=n, diseases='sis', networks=net, dur=ss.years(2), verbose=0).run() for k,net in nets.items()}
    degree = {k:np.bincount(np.r_[sim.networks[0].p1, sim.networks[0].p2], minlength=n) for k,sim in sims.items()}
    assert (degree['grid'] == 4).all() and (degree['small'].mean() == 4)
    assert np.isclose(degree['free'].mean(), 4, rtol=0.05) and degree['free'].max() > 3*degree['free'].mean() # A few have many contacts

    # Contacts across real places with a geography: as ssl.GeoNet makes them, a random network in a model stratified by place, with most contacts at home
    sis = ss.SIS().stratify(patch=ss.levels(ss.geo.geography('Kenya'), mobility=ss.geo.gravity(away=0.1)))
    sim = ss.Sim(n_agents=n, diseases=sis, networks=ss.RandomNet(n_contacts=6), dur=ss.years(2), verbose=0).run()
    net, place = sim.networks[0], sim.groups.level_raw(sim.contact_rates.space)
    assert (place[net.p1] == place[net.p2]).mean() > 0.5 # Mostly within a place
    geo = sims['geo'].networks[0]
    assert (geo.place.raw[geo.p1] == geo.place.raw[geo.p2]).mean() > 0.5 # Mostly within a place, as local=0.9 says
    return degree


@sc.timer()
def test_econ():
    """ Check that ssl.Costs adds up every cost, that ssl.discount() weights by time, that ssl.DALYs counts years of life, and that ssl.ICER compares scenarios with the baseline """
    sc.heading('Testing health economics...')

    S, I = ss.states('S I')
    infection = S >> I @ ss.peryear(2)*I
    recovery  = I >> S @ ss.years(1)
    sis = ss.Disease(infection, recovery, init_prev=0.1)
    vaccine = ss.vaccinate(efficacy=0.5, when=ss.schedule(every=ss.years(1), start=2005), coverage=0.5, cost=5.0)
    costs = ssl.Costs(unit_costs={'sis.infection': 0.1}, fixed=ss.peryear(1000), start=2005)
    base = ss.Sim(sis, analyzers=costs, total_pop=10_000, start=2000, stop=2020, verbose=0)
    msim = ss.parallel({'None': base, 'Vaccine': base.new(interventions=vaccine)}, workers=1)

    # The total is the sum of the delivery, care, and fixed costs
    sim = msim.sims['Vaccine']
    res = sim.results
    expected = res.vaccine.cost.sum() + 0.1*res.sis.infection.sum() + 1000*15
    assert np.isclose(res.costs.total.sum(), expected), f'Expected a total cost of {expected}, not {res.costs.total.sum()}'

    # Discounting leaves values before the start, and lowers those after it
    disc = ssl.discount(res.costs.total, 0.03, start=2005)
    assert np.allclose(disc.values[:5], res.costs.total.values[:5]), 'Values before the start should not be discounted'
    assert disc.sum() < res.costs.total.sum(), 'Discounting should lower the total'

    # The ICER is the incremental cost over the incremental outcome averted
    df = ssl.ICER(msim, outcome='sis.infection', baseline='None').df
    assert df.averted['None'] == 0 and df.averted['Vaccine'] > 0, 'Expected vaccination to avert infections'
    row = df.loc['Vaccine']
    assert np.isclose(row.ICER, (row.cost - df.cost['None'])/row.averted), 'Expected the ICER to be the incremental cost per infection averted'
    disc = ssl.ICER(msim, outcome='sis.infection', discount=0.03, discount_start=2005).df
    assert disc.cost['Vaccine'] < row.cost, 'Expected discounting to lower the cost'

    # A mix of two others can beat a scenario (extended dominance)
    icer = ssl.ICER()
    icer.t_years = np.zeros(1)
    for label,cost,outcome in [('A', 0, 100), ('B', 10, 95), ('C', 100, 90), ('D', 110, 50)]:
        icer.scenarios[label] = sc.obj(cost=np.array([[cost]], dtype=float), outcome=np.array([[outcome]], dtype=float))
    assert icer.df.extended['C'] and np.isnan(icer.df.ICER['C']) and icer.best(wtp=1000) == 'D'

    # Years of life lost: from ages with agents, or given; and compared while a sim runs
    pars = dict(diseases=ss.SIR(p_death=0.2, beta=ss.perday(0.3), dur_inf=ss.days(10)), networks='random', n_agents=1000, start=2000, dur=ss.years(1), dt=ss.days(7), verbose=0)
    make = lambda *extra: [ssl.Costs(unit_costs={'sir.infection': 1}), ssl.DALYs(weights={'sir.I': 0.1})] + list(extra)
    ref = ss.Sim(**pars, analyzers=make()).run()
    dalys = ref.results.dalys
    assert 30 < dalys.yll.sum()/ref.results.sir.death.sum() < 89, 'Expected the years of life lost per death to be from the life table'
    assert np.isclose(dalys.dalys.sum(), dalys.yld.sum() + dalys.yll.sum())

    return df


def malaria(): # The square root of the classical Ross-Macdonald value, m·a²·b·c/(γ·μ_v), since a vector doubles the generations
    dis = ssl.Malaria()
    sim = ss.Sim(dis, total_pop=10_000, unit='day', stop=400, verbose=0).run()
    assert np.isclose(dis.R0(), np.sqrt(10.09*0.5**2*0.5*0.8/(0.033*0.143)), rtol=0.01)
    assert sim.results.malaria.prevalence.max() > 0.5, 'Expected malaria to take off'


def malaria_immunity(): # Malaria dies out below the critical vectorial capacity, (alpha1 + delta)/g = 0.022 per day
    prev = lambda C: ss.Sim(ssl.MalariaImmunity(C=C), total_pop=10_000, unit='day', dt=ss.days(10), dur=ss.days(365*20), verbose=0).run().results.malariaimmunity
    assert prev(0.011).prevalence[-1] < 0.001, 'Expected malaria to die out below the critical vectorial capacity'
    above = prev(0.044)
    assert above.prevalence[-1] > 0.1, 'Expected malaria to be endemic above the critical vectorial capacity'
    assert above.detected[-1] < above.prevalence[-1], 'Expected immune people to be detected less often'


def tb(): # The LSHTM reference, in which TB declines slowly over centuries; the population is held constant by replacing TB deaths
    sim = ss.Sim(ssl.TB(init=dict(Y=0.01)), solver='ode', total_pop=100_000, start=1900, stop=2100, dt=ss.years(1), verbose=0).run()
    res = sim.results.tb
    assert np.isclose(res.prevalence[100], 0.0025, atol=0.0005), 'Expected the prevalence of the reference ODE in 2000'
    assert np.isclose(sum(res[k][-1] for k in ['S', 'L', 'C', 'R', 'N', 'A', 'Y']), 100_000, rtol=0.01), 'Expected a constant population'


def covid():
    sim = ss.Sim(ssl.COVID(), total_pop=83_000, unit='day', stop=120, verbose=0).run()
    res = sim.results.covid
    assert 2.5 < sim.diseases.covid.R0() < 4, 'Expected an R0 of about 3.2 with MEmilio\'s values by age'
    deaths = res.outcome.select(to='dead').by('age').values.sum(0)
    assert np.argmax(deaths) == 4 and deaths[4] > 10*deaths[2], 'Expected most deaths at 60-79, far more than among young adults'
    simple = ss.Sim(ssl.COVID(ages=False), total_pop=83_000, unit='day', stop=120, verbose=0).run()
    assert 2 < simple.diseases.covid.R0() < 3.5, 'Expected an R0 of about 2.7 for the simple MEmilio example'
    assert simple.results.covid.ICr.max() > 0 and simple.results.covid.outcome.sum() > 0, 'Expected critical cases and deaths'


def influenza(): # Transmission peaks each January, so epidemics return each winter; deaths are mostly among the over-65s
    pars = dict(total_pop=100_000, start='2000-01-01', stop='2006-01-01', dt=ss.days(1), verbose=0)
    sim = ss.Sim(ssl.Influenza(), **pars).run()
    res = sim.results.influenza
    assert 1.5 < sim.diseases.influenza.R0() < 1.9, 'Expected an R0 of 1.3 on average, and 1.7 at the January peak'
    yearly = np.array([res.infection[i*365:(i + 1)*365].sum() for i in range(6)])
    assert (yearly > 0.1*100_000).all() and (yearly < 0.4*100_000).all(), 'Expected influenza to return each year, infecting 10-40% of people'
    deaths = res.outcome.select(to='dead').by('age').values.sum(0)
    assert np.argmax(deaths) == 6 and deaths[6] > deaths[:6].sum(), 'Expected most deaths over 65'
    vax = ss.Sim(ssl.Influenza(cov_vax=ss.by(age=[0, 0.5, 0.5, 0, 0, 0, 0.7])), **pars).run()
    assert vax.results.influenza.vaccination.sum() > 0 and vax.results.influenza.infection.sum() < 0.9*res.infection.sum(), 'Expected vaccination to prevent infections'
    assert vax.results.influenza.outcome.select(to='dead').sum() < 0.85*res.outcome.select(to='dead').sum(), 'Expected vaccinating the over-65s to prevent deaths'


def rsv(): # Repeated infections; first infections in babies are the severe ones, and immunizing babies prevents some
    assert 8 < ssl.RSV().R0() < 11, 'Expected the R0 of about 9 of Pitzer et al.'
    kw = dict(total_pop=100_000, start='2000-01-01', stop='2008-01-01', dt=ss.days(1), verbose=0)
    sim = ss.Sim(ssl.RSV(), **kw).run()
    res = sim.results.rsv
    assert res.M[-1] > 0 and res.maternal.sum() > 0, 'Expected babies to be born protected and to lose that protection'
    assert res.infection4.sum() > res.infection1.sum(), 'Expected most infections to be reinfections'
    hosp = res.recovery1.select(to='H1').by('age').values[-3*365:].sum(0)
    assert np.argmax(hosp) <= 2, 'Expected most hospitalizations with a first infection under 6 months'
    assert res.recovery4.select(to='H4').by('age').values[:, -1].sum() > 0, 'Expected older adults (65+) to be hospitalized'
    immunized = ss.Sim(ssl.RSV(cov_mab=0.8), **kw).run()
    hosp_imm = immunized.results.rsv.recovery1.select(to='H1').by('age').values[-3*365:].sum(0)
    assert hosp_imm[:3].sum() < 0.9*hosp[:3].sum(), 'Expected nirsevimab to prevent hospitalizations under 6 months'


def pertussis(): # Exposure boosts waned immunity; the DTP series prevents infant deaths and moves infections to older ages
    pars = dict(total_pop=100_000, start=1920, stop=1980, dt=ss.days(5), verbose=0)
    sims = [ss.Sim(ssl.Pertussis(cov_dtp=cov), **pars).run() for cov in [0, 0.8]]
    assert 11 < sims[0].diseases.pertussis.R0() < 15, 'Expected the R0 of 11 to 15 estimated by Wearing and Rohani'
    pre, dtp = [sim.results.pertussis for sim in sims]
    last = slice(-20*73, None) # The last 20 years
    infants = lambda res: res.infection.by('age').values[last, :4].sum()/res.infection.values[last].sum()
    assert pre.boosting.sum() > pre.infection.sum() > 0 and pre.loss.sum() > 0, 'Expected immunity to wane, and to be boosted more often than people are infected'
    assert dtp.outcome.dead[last].sum() < 0.5*pre.outcome.dead[last].sum(), 'Expected the DTP series to prevent most deaths'
    assert infants(dtp) < 0.7*infants(pre), 'Expected vaccination to move infections out of infancy'


def typhoid(): # Pitzer et al.'s R0 of 2.8, to which carriers add; routine TCV lowers first infections; deaths by age
    mean = dict(beta_w=ss.perweek(0.0175)) # The seasonal rate from the water, at its mean
    assert np.isclose(ssl.Typhoid(**mean).R0(), 2.8, atol=0.1), 'Expected the R0 of 2.8 estimated by Pitzer et al.'
    assert ssl.Typhoid(p_carrier=0, **mean).R0() < ssl.Typhoid(**mean).R0(), 'Expected the carriers to add to R0'
    pars = dict(total_pop=20_000, start=1960, stop=2020, dt=ss.weeks(2), verbose=0)
    base, vax = [ss.Sim(ssl.Typhoid(cov_tcv=cov), **pars).run().results.typhoid for cov in [0, 0.8]]
    assert base.C[-1] > 0 and base.V[-1] == 0 < vax.V[-1], 'Expected chronic carriers, and the vaccinated only with TCV'
    assert vax.infection[-260:].sum() < 0.9*base.infection[-260:].sum(), 'Expected routine TCV to lower first infections'
    assert base.recovery.select(to='dead').by('age').values.sum() > 0, 'Expected deaths from typhoid, by age'


def polio(): # Paralysis comes from first infections, mostly under 5; OPV prevents it, and its virus spreads
    pars = dict(total_pop=1_000_000, start=1950, stop=1980, dt=ss.days(5), verbose=0)
    sim = ss.Sim(ssl.Polio(), **pars).run()
    res = sim.results.polio
    assert 7.5 < sim.diseases.polio.R0() < 8.5, 'Expected the R0 of 8 for type 1 of Kalkowska et al.'
    P = res.infection.P.by('age').values[-730:].sum(0) # The last 10 years
    assert P[:4].sum() > 0.8*P.sum() and P[0] < P[1], 'Expected most paralysis under 5, but little under 6 months'
    assert 1/300 < res.infection.P.sum()/res.infection.sum() < 1/190, 'Expected about 1 paralytic case per 190 first infections, fewer in babies'
    vax = ss.Sim(ssl.Polio(cov_opv=0.8), **pars).run().results.polio
    assert vax.infection.P.sum() < 0.5*res.infection.P.sum() and vax.opv_spread.sum() > 0, 'Expected OPV to prevent paralysis, and its virus to spread'
    agents = ss.Sim(ssl.Polio(cov_opv=0.8, cov_ipv=0.5), n_agents=1000, start=1950, stop=1955, dt=ss.days(10), verbose=0).run().results.polio
    assert agents.routine.O.sum() > 0 and agents.routine.V.sum() > 0, 'Expected vaccination as agents'


def sti(): # Both run on a sexual network; syphilis's modifiers lower each latent person's infectiousness
    net = lambda: ss.SexualNet(dur=ss.years(2), n_contacts=ss.poisson(1))
    pars = dict(n_agents=1000, start=2000, stop=2020, dt=ss.months(2), rand_seed=1, verbose=0)
    syph = ss.Sim(diseases=ssl.Syphilis(), networks=net(), **pars).run()
    dis = syph.diseases.syphilis
    latent = dis.El.uids
    rel_trans = dis.current_value('rel_trans', latent)
    assert len(latent) and rel_trans.max() <= 1 and rel_trans.min() < 0.9, 'Expected latent infectiousness to fall with time'
    gono = ss.Sim(diseases=ssl.Gonorrhea(), networks=net(), **pars).run()
    assert gono.results.gonorrhea.prevalence[-1] > 0, 'Expected gonorrhea to persist'


@sc.timer()
def measles(): # R0 is beta times the infectious period; deaths are by age; and each dose protects its coverage times its efficacy, as compartments and as agents
    assert np.isclose(ssl.Measles().R0(), 1.4*11, rtol=0.01) # Less the few who die of other causes while infected
    outbreak = dict(start='2025-01-01', stop='2025-07-01', rand_seed=1, verbose=0)
    res = ss.Sim(diseases=ssl.Measles(), total_pop=10_000, **outbreak).run().results.measles
    died, recovered = res.death.by('age').values.sum(axis=0), res.recovery.by('age').values.sum(axis=0)
    assert res.infection.sum() > 9000 and np.allclose((died/(died + recovered))[[0, -1]], [0.006, 0.0013], rtol=0.01) # p_death in infants and adults
    res = ss.Sim(diseases=ssl.Measles(p_death=0.1), n_agents=2000, **outbreak).run().results.measles
    assert res.infection.sum() > 1800 and np.isclose(res.death.sum()/(res.death.sum() + res.recovery.sum()), 0.1, rtol=0.3)

    # Routine vaccination, with no transmission: a share cov*eff of the children who reach each dose are protected
    for size, rtol in [(dict(total_pop=10_000), 0.001), (dict(n_agents=2000), 0.25)]:
        vaccinated = ssl.Measles(beta=ss.perday(0), cov_mcv1=0.8, cov_mcv2=0.5, init=dict(I=0))
        res = ss.Sim(diseases=vaccinated, start='2000-01-01', stop='2010-01-01', dt=ss.days(7), rand_seed=1, verbose=0, **size).run().results.vaccinated
        share = [dose.select(to='V').sum()/dose.sum() for dose in [res.mcv1, res.mcv2]]
        assert np.allclose(share, [0.8*0.84, 0.5*0.925], rtol=rtol), (size, share)

    # An instance takes extra states, transitions, and outcomes beside the class's, which stand for the instance's copies
    H = ss.State('Hospitalized')
    hosp = ssl.Measles(H, admission=ss.Transition(ssl.Measles.I, H, rate=ss.perday(0.01)), cases=ss.outcome(0.5*ssl.Measles.onset))
    res = ss.Sim(diseases=hosp, total_pop=10_000, start='2025-01-01', stop='2025-04-01', verbose=0).run().results.hosp
    assert res.admission.sum() > 0 and np.isclose(res.cases.sum(), 0.5*res.onset.sum()) and 'H' not in ssl.Measles().graph.states


@sc.timer()
def cholera(): # The water carries the epidemic: without it, transmission from person to person alone dies out
    pars = dict(total_pop=10_000, start='2025-01-01', stop='2026-01-01', verbose=0)
    water = ss.Sim(diseases=ssl.Cholera(), **pars).run().results.cholera
    direct = ss.Sim(diseases=ssl.Cholera(beta_env=0), **pars).run().results.cholera
    assert water.infection.sum() > 5000 and direct.infection.sum() < 100 and water.W.max() > 1000 and water.env_conc.max() > 0.01


@sc.timer()
def ebola(): # Safe burials prevent infections from bodies; every death is buried
    pars = dict(total_pop=10_000, start='2025-01-01', stop='2026-01-01', verbose=0)
    some, every = [ss.Sim(diseases=ssl.Ebola(p_safe_bury=p), **pars).run().results.ebola for p in [0.25, 1.0]]
    assert every.infection.sum() < some.infection.sum() and every.D.max() < 1e-9 < some.D.max()
    assert np.isclose(some.deaths.sum(), some.burial.sum() + some.D[-1], rtol=0.01) and some.Sev.max() > 0


declared = dict(measles=measles, cholera=cholera, ebola=ebola, malaria=malaria, malaria_immunity=malaria_immunity, tb=tb, covid=covid, influenza=influenza,
                rsv=rsv, pertussis=pertussis, typhoid=typhoid, polio=polio, sti=sti)


@pytest.mark.parametrize('key', declared.keys())
def test_declared_diseases(key):
    """ Check each declared library disease against a value from the source it came from, or its headline mechanism """
    sc.heading(f'Testing {key}...')
    with sc.timer(key):
        declared[key]()
    return


if __name__ == '__main__':
    test_hiv()
    test_networks()
    test_econ()
    for _key in declared:
        test_declared_diseases(_key)
