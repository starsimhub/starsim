"""
Run tests of the Starsim library (starsim.library, i.e. ssl)

These tests check that every library module can be instantiated and run, and that
its headline mechanism actually does something. Detailed tests of individual
modules live in the corresponding test files (e.g. test_diseases.py).
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import pytest
import starsim as ss
import starsim.library as ssl

n_agents = 2_000

# Outbreak diseases specify their natural history in days, so they need a sub-year timestep
outbreak_pars = dict(start=2000, dur=ss.years(1), dt=ss.days(1), n_agents=n_agents, rand_seed=1, verbose=0)
endemic_pars  = dict(start=2000, stop=2030, n_agents=n_agents, rand_seed=1, verbose=0)
mnch_pars     = dict(start=2000, dur=ss.years(3), dt=ss.weeks(1), n_agents=n_agents, rand_seed=1, verbose=0)


@sc.timer()
def test_diseases():
    """
    Check that each library disease runs and produces infections, and that the SEIR-type ones (written as code) keep consistent compartments

    'exposed' and 'infectious' are the literal E and I compartments, while 'infected' is
    derived as E plus I. Infections are counted when they happen rather than inferred from
    a ti_ state, so seed infections are excluded exactly once. Regression test for negative
    first-step incidence (issue #1389).
    """
    sc.heading('Testing library diseases and their compartments...')

    diseases = dict(
        agentseir = ssl.AgentSEIR(beta=ss.perday(0.3), init_prev=0.1),
        cholera = ssl.Cholera(beta=ss.perday(0.5), init_prev=0.1),
        ebola   = ssl.Ebola(beta=ss.perday(0.1), init_prev=0.1),
        measles = ssl.Measles(beta=ss.perday(0.3), init_prev=0.1),
    )
    sims = sc.obj()
    for name, disease in diseases.items():
        sim = ss.Sim(diseases=disease, networks=ss.RandomNet(), **outbreak_pars)
        sim.run()
        sims[name] = sim
        dis = sim.diseases[name]
        res = sim.results[name]
        n_exp = np.array(res.exposed)
        n_infectious = np.array(res.infectious)
        n_inf = np.array(res.infected)
        new = np.array(res.infection)
        total = n_exp + n_infectious + np.array(res.susceptible) + np.array(res.recovered)
        assert res.infection.cum[-1] > 0, f'Expected {name} infections'
        assert new.min() >= 0, f'{name} has negative infection: seeds are being double-counted'
        assert new[0] == 0, f'{name} counted its seed infections as new infections'
        assert new.sum() == res.infection.cum[-1], f'{name} cumulative infections do not match the sum'
        assert not (dis.exposed & dis.infectious).any(), f'{name} has agents in both E and I'
        assert np.array_equal(total, np.array(sim.results.alive)), f'{name} compartments do not sum to the population'
        assert np.array_equal(n_inf, n_exp + n_infectious), f'{name} should count both E and I as infected'
        assert (dis.infectious & ~dis.infected).sum() == 0, f'{name} should transmit only from agents that are infected'
        assert n_exp.max() > 0, f'{name} never had any exposed agents'
        assert n_inf.max() > 0, f'{name} never had any infectious agents'

    # Cholera should also track its environmental reservoir
    assert sims.cholera.results.cholera.env_prev.max() > 0, 'Expected nonzero cholera environmental prevalence'

    # Ebola should progress agents to severe disease and bury the dead
    ebola = sims.ebola.results.ebola
    assert ebola.severe.max() > 0, 'Expected some severe Ebola cases'
    assert ebola.buried.max() > 0, 'Expected some Ebola burials'

    # Cholera dies out completely with a yearly timestep, since its natural history is in days
    yearly = ss.Sim(diseases=ssl.Cholera(beta=ss.perday(0.5)), networks=ss.RandomNet(),
                    start=2000, dur=ss.years(1), dt=ss.years(1), n_agents=n_agents, rand_seed=1, verbose=0)
    yearly.run()
    assert yearly.results.cholera.infection.cum[-1] == 0, 'Cholera unexpectedly worked with a yearly timestep'
    return sims


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
    s2 = make('With ART', ssl.ART(year=[2005, 2015], coverage=[0, 0.9]))
    for sim in [s1, s2]:
        sim.run()

    d1 = s1.results.hiv.death.sum()
    d2 = s2.results.hiv.death.sum()
    n_art = s2.results.art.n_art.max()
    print(f'  Deaths without ART: {d1:n}; with ART: {d2:n}; peak on ART: {n_art:n}')
    assert n_art > 0, 'Expected some agents on ART'
    assert d2 < d1, 'Expected ART to reduce HIV deaths'

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
    nets = dict(geo=ssl.GeoNet('Kenya', n_contacts=6), grid=ssl.GridNet(periodic=True), small=ssl.SmallWorldNet(k=4), free=ssl.ScaleFreeNet(m=2))
    sims = {k:ss.Sim(n_agents=n, diseases='sis', networks=net, dur=2, verbose=0).run() for k,net in nets.items()}
    degree = {k:np.bincount(np.r_[sim.networks[0].p1, sim.networks[0].p2], minlength=n) for k,sim in sims.items()}
    assert (degree['grid'] == 4).all() and (degree['small'].mean() == 4)
    assert degree['free'].max() > 3*degree['free'].mean() # A few have many contacts
    geo = sims['geo'].networks[0]
    assert np.isclose(degree['geo'].mean(), 6, rtol=0.05)
    assert (geo.place.raw[geo.p1] == geo.place.raw[geo.p2]).mean() > 0.5 # Mostly within a place
    return degree


@sc.timer()
def test_econ():
    """ Check that ssl.Costs adds up every cost, that ssl.discount() weights by time, that ssl.DALYs counts years of life, and that ssl.ICER compares scenarios with the baseline """
    sc.heading('Testing health economics...')

    S, I = ss.states('S I')
    infection = S >> I @ ss.peryear(2)*I
    recovery  = I >> S @ ss.years(1)
    sis = ss.Disease(infection, recovery, init_prev=0.1)
    vaccine = ss.vaccinate(efficacy=0.5, coverage=0.5, interval=ss.years(1), start=2005, cost=5.0)
    costs = ssl.Costs(unit_costs={'sis.infection': 0.1}, fixed=1000, start=2005)
    base = ss.Sim(sis, analyzers=costs, total_pop=10_000, start=2000, stop=2020, verbose=0)
    msim = ss.parallel({'None': base, 'Vaccine': base.new(interventions=vaccine)}, parallel=False)

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
    print(df)
    assert df.averted['None'] == 0 and df.averted['Vaccine'] > 0, 'Expected vaccination to avert infections'
    row = df.loc['Vaccine']
    assert np.isclose(row.ICER, (row.cost - df.cost['None'])/row.averted), 'Expected the ICER to be the incremental cost per infection averted'

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
    live = ss.Sim(**pars, label='Live', analyzers=make(ssl.ICER(dict(Reference=ref)))).run()
    assert set(live.analyzers.icer.df.index) == {'Reference', 'Live'}
    with pytest.raises(ValueError, match='life_lost'):
        ss.Sim(sis, analyzers=ssl.DALYs(), total_pop=1000, verbose=0).init()

    return df


def malaria(): # The square root of the classical Ross-Macdonald value, m·a²·b·c/(γ·μ_v), since a vector doubles the generations
    dis = ssl.Malaria()
    sim = ss.Sim(dis, total_pop=10_000, unit='day', stop=400, verbose=0).run()
    assert np.isclose(dis.R0(), np.sqrt(10.09*0.5**2*0.5*0.8/(0.033*0.143)), rtol=0.01)
    assert sim.results.malaria.prevalence.max() > 0.5, 'Expected malaria to take off'


def malaria_immunity(): # Malaria dies out below the critical vectorial capacity, (alpha1 + delta)/g = 0.022 per day
    prev = lambda C: ss.Sim(ssl.MalariaImmunity(C=C), total_pop=10_000, unit='day', dt=ss.days(10), stop=365*20, verbose=0).run().results.malariaimmunity
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
    assert 2 < sim.diseases.covid.R0() < 3.5, 'Expected an R0 of about 2.7 for the MEmilio example'
    assert res.ICr.max() > 0 and res.outcome.sum() > 0, 'Expected critical cases and deaths'


def influenza(): # Transmission peaks each January, so epidemics return each winter
    sim = ss.Sim(ssl.Influenza(), total_pop=100_000, start='2000-01-01', stop='2006-01-01', dt=ss.days(1), verbose=0).run()
    yearly = np.array([sim.results.influenza.infection[i*365:(i + 1)*365].sum() for i in range(6)])
    assert (yearly[4:] > 0.05*100_000).all(), 'Expected influenza epidemics to keep returning'


def rsv(): # Babies are born with maternal antibodies, then become susceptible
    sim = ss.Sim(ssl.RSV(), total_pop=100_000, start='2000-01-01', stop='2004-01-01', dt=ss.days(1), verbose=0).run()
    res = sim.results.rsv
    assert res.M[-1] > 0 and res.maternal.sum() > 0, 'Expected babies to be born protected and to lose that protection'
    assert res.prevalence[-1] > 0, 'Expected RSV to persist'


def pertussis():
    assert 11 < ssl.Pertussis().R0() < 15, 'Expected the R0 of 11 to 15 estimated by Wearing and Rohani'
    sim = ss.Sim(ssl.Pertussis(), total_pop=100_000, start='1920-01-01', stop='1960-01-01', dt=ss.days(5), verbose=0).run()
    assert sim.results.pertussis.waning.sum() > 0, 'Expected immunity to wane'


def dengue(): # Having had one serotype protects in part against the others, so there are fewer infections
    run = lambda cross: ss.Sim(diseases=ssl.Dengue(cross=cross), n_agents=500, unit='day',
                                 stop=365, rand_seed=1, verbose=0).run().results
    total = lambda cross: sum(res.infection.sum() for res in [run(cross)[f'den{i + 1}'] for i in range(4)])
    assert total(1.0) < 0.8*total(0.0), 'Expected cross-protection to reduce the infections'


def typhoid(): # Carriers are a hundredth as infectious, but shed for life, so they matter for R0
    assert np.isclose(ssl.Typhoid().R0(), 2.8, atol=0.1), 'Expected the R0 of 2.8 estimated by Pitzer et al.'
    assert ssl.Typhoid(p_carrier=0).R0() < ssl.Typhoid().R0(), 'Expected the carriers to add to R0'
    sim = ss.Sim(ssl.Typhoid(), total_pop=20_000, start=2000, stop=2040, dt=ss.weeks(4), verbose=0).run()
    assert sim.results.typhoid.C[-1] > 0, 'Expected chronic carriers'


def polio(): # About 1 infection in 200 causes paralysis
    sim = ss.Sim(ssl.Polio(), total_pop=100_000, start=1980, stop=1990, dt=ss.days(5), verbose=0).run()
    res = sim.results.polio
    assert np.isclose(res['outcome.P'].sum()/res.onset.sum(), 1/200, rtol=0.2), 'Expected about 1 paralytic case per 200 infections'


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


declared = dict(malaria=malaria, malaria_immunity=malaria_immunity, tb=tb, covid=covid, influenza=influenza,
                rsv=rsv, pertussis=pertussis, dengue=dengue, typhoid=typhoid, polio=polio, sti=sti)


@pytest.mark.parametrize('key', declared.keys())
def test_declared_diseases(key):
    """ Check each declared library disease against a value from the source it came from, or its headline mechanism """
    sc.heading(f'Testing {key}...')
    with sc.timer(key):
        declared[key]()
    return


if __name__ == '__main__':
    diseases = test_diseases()
    hiv      = test_hiv()
    econ     = test_econ()
    for _key in declared:
        test_declared_diseases(_key)
