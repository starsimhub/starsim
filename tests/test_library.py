"""
Run tests of the Starsim library (starsim.library, i.e. ssl)

These tests check that every library module can be instantiated and run, and that
its headline mechanism actually does something. Detailed tests of individual
modules live in the corresponding test files (e.g. test_diseases.py).
"""

# %% Imports and settings
import inspect
import sciris as sc
import numpy as np
import starsim as ss
import starsim.library as ssl

test_run = True
n_agents = [10_000, 2_000][test_run]
do_plot = True
sc.options(interactive=do_plot) # Assume not running interactively

# Outbreak diseases specify their natural history in days, so they need a sub-year timestep
outbreak_pars = dict(start=2000, dur=ss.years(1), dt=ss.days(1), n_agents=n_agents, rand_seed=1, verbose=0)
endemic_pars  = dict(start=2000, stop=2030, n_agents=n_agents, rand_seed=1, verbose=0)
mnch_pars     = dict(start=2000, dur=ss.years(5), dt=ss.weeks(1), n_agents=n_agents, rand_seed=1, verbose=0)


@sc.timer()
def test_exports():
    """ Check that every library class is importable from both the top level and its submodule """
    sc.heading('Testing library exports...')

    for submod in [ssl.diseases, ssl.networks, ssl.mnch]:
        for name,obj in vars(submod).items():
            if name.startswith('_') or inspect.ismodule(obj): # Skip dunders and submodules
                continue
            assert getattr(ssl, name, None) is obj, f'ssl.{name} is missing or differs from {submod.__name__}.{name}'
            assert name in ssl.__all__, f'{name} is missing from ssl.__all__'

    sc.printgreen('✓ Library export tests passed')
    return ssl.__all__


@sc.timer()
def test_diseases():
    """ Check that each library disease runs and produces infections """
    sc.heading('Testing library diseases...')

    diseases = dict(
        cholera = ssl.Cholera(beta=ss.perday(0.5)),
        ebola   = ssl.Ebola(beta=ss.perday(0.1)),
        measles = ssl.Measles(beta=ss.perday(0.3)),
    )
    sims = sc.objdict()
    for name, disease in diseases.items():
        sim = ss.Sim(diseases=disease, networks=ss.RandomNet(), **outbreak_pars)
        sim.run()
        res = sim.results[name]
        print(f'  {name}: {res.cum_infections[-1]:n} cumulative infections')
        assert res.cum_infections[-1] > 0, f'Expected {name} infections'
        sims[name] = sim

    # Cholera should also track its environmental reservoir
    env = sims.cholera.results.cholera.env_prev
    assert env.max() > 0, 'Expected nonzero cholera environmental prevalence'

    # Ebola should progress agents to severe disease and bury the dead
    ebola = sims.ebola.results.ebola
    assert ebola.n_severe.max() > 0, 'Expected some severe Ebola cases'
    assert ebola.n_buried.max() > 0, 'Expected some Ebola burials'

    # Cholera dies out completely with a yearly timestep, since its natural history is in days
    yearly = ss.Sim(diseases=ssl.Cholera(beta=ss.perday(0.5)), networks=ss.RandomNet(),
                    start=2000, dur=ss.years(1), n_agents=n_agents, rand_seed=1, verbose=0)
    yearly.run()
    assert yearly.results.cholera.cum_infections[-1] == 0, 'Cholera unexpectedly worked with a yearly timestep'

    sc.printgreen('✓ Library disease tests passed')
    return sims


@sc.timer()
def test_compartments():
    """
    Check the SEIR-type library diseases keep consistent compartments

    'exposed' and 'infectious' are the literal E and I compartments, while 'infected' is
    derived as E plus I. Infections are counted when they happen rather than inferred from
    a ti_ state, so seed infections are excluded exactly once. Regression test for negative
    first-step incidence (issue #1389).
    """
    sc.heading('Testing disease compartments...')

    diseases = dict(
        seir    = ss.SEIR(beta=ss.perday(0.3), init_prev=0.1),
        cholera = ssl.Cholera(beta=ss.perday(0.5), init_prev=0.1),
        ebola   = ssl.Ebola(beta=ss.perday(0.1), init_prev=0.1),
        measles = ssl.Measles(beta=ss.perday(0.3), init_prev=0.1),
    )
    for name, disease in diseases.items():
        sim = ss.Sim(diseases=disease, networks=ss.RandomNet(), **outbreak_pars)
        sim.run()
        dis = sim.diseases[name]
        res = sim.results[name]
        n_exp = np.array(res.n_exposed)
        n_infectious = np.array(res.n_infectious)
        n_inf = np.array(res.n_infected)
        new = np.array(res.new_infections)
        total = n_exp + n_infectious + np.array(res.n_susceptible) + np.array(res.n_recovered)
        print(f'  {name}: new_infections[0]={new[0]:n}, min={new.min():n}')

        assert new.min() >= 0, f'{name} has negative new_infections: seeds are being double-counted'
        assert new[0] == 0, f'{name} counted its seed infections as new infections'
        assert new.sum() == res.cum_infections[-1], f'{name} cumulative infections do not match the sum'
        assert not (dis.exposed & dis.infectious).any(), f'{name} has agents in both E and I'
        assert np.array_equal(total, np.array(sim.results.n_alive)), f'{name} compartments do not sum to the population'
        assert np.array_equal(n_inf, n_exp + n_infectious), f'{name} should count both E and I as infected'
        assert (dis.infectious & ~dis.infected).sum() == 0, f'{name} should transmit only from agents that are infected'
        assert n_exp.max() > 0, f'{name} never had any exposed agents'
        assert n_inf.max() > 0, f'{name} never had any infectious agents'

    sc.printgreen('✓ Disease compartment tests passed')
    return


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

    d1 = s1.results.hiv.new_deaths.sum()
    d2 = s2.results.hiv.new_deaths.sum()
    n_art = s2.results.art.n_art.max()
    print(f'  Deaths without ART: {d1:n}; with ART: {d2:n}; peak on ART: {n_art:n}')
    assert n_art > 0, 'Expected some agents on ART'
    assert d2 < d1, 'Expected ART to reduce HIV deaths'

    # The analyzer should have recorded plausible CD4 counts
    cd4 = s1.analyzers.cd4_analyzer.cd4
    assert cd4.shape[0] == s1.t.npts, 'Expected one CD4 row per timestep'
    assert 0 < cd4.mean() <= 500, f'Expected CD4 counts in (0, 500], not {cd4.mean()}'

    sc.printgreen('✓ HIV tests passed')
    return s1, s2


@sc.timer()
def test_networks():
    """ Check that each library network runs and forms the expected edges """
    sc.heading('Testing library networks...')

    sims = sc.objdict()
    for name, net in dict(disk=ssl.DiskNet(r=0.05), er=ssl.ErdosRenyiNet(p=0.01)).items():
        sim = ss.Sim(diseases=ss.SIS(beta=ss.perday(0.2)), networks=net, **outbreak_pars)
        sim.run()
        n_edges = len(sim.networks[0])
        print(f'  {name}: {n_edges:n} edges, {sim.results.sis.cum_infections[-1]:n} cumulative infections')
        assert n_edges > 0, f'Expected edges in {name} network'
        assert sim.results.sis.cum_infections[-1] > 0, f'Expected infections via {name} network'
        sims[name] = sim

    # NullNet has one zero-weight self-edge per agent, so nothing should transmit
    sim = ss.Sim(diseases=ss.SIS(beta=ss.perday(0.2)), networks=ssl.NullNet(), **outbreak_pars)
    sim.run()
    net = sim.networks.nullnet
    assert len(net) == n_agents, 'Expected one self-edge per agent in NullNet'
    assert np.all(net.edges.beta == 0), 'Expected zero transmission weight in NullNet'
    assert (net.edges.p1 == net.edges.p2).all(), 'Expected only self-edges in NullNet'
    sims.null = sim

    sc.printgreen('✓ Library network tests passed')
    return sims


@sc.timer()
def test_households():
    """ Check that HouseholdNet assigns plausible households """
    sc.heading('Testing HouseholdNet...')

    sim = ss.Sim(diseases='sis', networks=ssl.HouseholdNet(dhs_data='default'),
                 demographics=ss.Pregnancy(), **endemic_pars)
    sim.run()

    net = sim.networks.householdnet
    hh_ids = net.household_ids[net.household_ids.notnan] # Unborn agents have no household
    sizes = np.bincount(hh_ids.astype(int))
    sizes = sizes[sizes > 0]
    print(f'  Households: {len(sizes):n}; mean size: {sizes.mean():0.2f}; max size: {sizes.max():n}')
    assert len(sizes) > 1, 'Expected more than one household'
    assert sizes.min() >= 1 and sizes.mean() > 1, 'Expected households with more than one member'
    assert len(net) > 0, 'Expected household edges'

    sc.printgreen('✓ HouseholdNet tests passed')
    return sim


@sc.timer()
def test_mnch():
    """ Check that the MNCH modules run and produce their headline outcomes """
    sc.heading('Testing MNCH modules...')

    demographics = lambda: [ss.Pregnancy(fertility_rate=ss.freqperyear(100), burnin=True), ss.Deaths()]
    networks = lambda: [ss.PrenatalNet(), ss.RandomNet()]
    sims = sc.objdict()

    # Congenital disease: mother-to-child transmission with birth outcomes
    sim = ss.Sim(diseases=ssl.mnch.CongenitalDisease(beta=ss.peryear(20), init_prev=0.2),
                 demographics=demographics(), networks=networks(), **mnch_pars)
    sim.run()
    n_congenital = sim.diseases.congenitaldisease.congenital.sum()
    print(f'  Congenital infections: {n_congenital:n}')
    assert n_congenital > 0, 'Expected some congenital infections'
    sims.congenital = sim

    # Neonatal sepsis: newborns are infected at birth, and some die
    sim = ss.Sim(diseases=ssl.mnch.NeonatalSepsis(), demographics=demographics(),
                 networks=networks(), **mnch_pars)
    sim.run()
    new_inf = sim.results.neonatalsepsis.new_infections.sum()
    nnd = sim.results.pregnancy.neonatal_deaths.sum()
    print(f'  Neonatal sepsis infections: {new_inf:n}; neonatal deaths: {nnd:n}')
    assert new_inf > 0, 'Expected newborns to be infected with sepsis'
    assert nnd > 0, 'Expected some neonatal deaths'
    sims.sepsis = sim

    # Fetal health: infection during pregnancy should restrict growth
    sim = ss.Sim(diseases=ss.SIR(beta=ss.peryear(20)), demographics=demographics(),
                 connectors=ssl.mnch.fetal_infection(), custom=ssl.mnch.FetalHealth(),
                 interventions=ssl.mnch.treat_pregnant(disease='sir', start_year=2003),
                 networks=networks(), **mnch_pars)
    sim.run()
    fh = sim.custom.fetal_health
    n_treated = sim.interventions.treat_pregnant.ti_treated.notnan.sum()
    n_weighed = fh.birth_weight.notnan.sum()
    print(f'  Newborns weighed: {n_weighed:n}; LBW: {fh.lbw.sum():n}; treated: {n_treated:n}')
    assert n_weighed > 0, 'Expected some newborns to have a birth weight'
    assert n_treated > 0, 'Expected some pregnant women to be treated'
    sims.fetal = sim

    sc.printgreen('✓ MNCH tests passed')
    return sims


if __name__ == '__main__':
    exports  = test_exports()
    diseases = test_diseases()
    comps    = test_compartments()
    hiv      = test_hiv()
    networks = test_networks()
    hh       = test_households()
    mnch     = test_mnch()
