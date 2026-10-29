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

n_agents = 2_000

# Outbreak diseases specify their natural history in days, so they need a sub-year timestep
outbreak_pars = dict(start=2000, dur=ss.years(1), dt=ss.days(1), n_agents=n_agents, rand_seed=1, verbose=0)
endemic_pars  = dict(start=2000, stop=2030, n_agents=n_agents, rand_seed=1, verbose=0)
mnch_pars     = dict(start=2000, dur=ss.years(3), dt=ss.weeks(1), n_agents=n_agents, rand_seed=1, verbose=0)


@sc.timer()
def test_exports():
    """ Check that every library class is importable from both the top level and its submodule """
    sc.heading('Testing library exports...')

    for submod in [ssl.diseases, ssl.networks, ssl.mnch, ssl.econ]:
        for name,obj in vars(submod).items():
            if name.startswith('_') or inspect.ismodule(obj): # Skip dunders and submodules
                continue
            assert getattr(ssl, name, None) is obj, f'ssl.{name} is missing or differs from {submod.__name__}.{name}'
            assert name in ssl.__all__, f'{name} is missing from ssl.__all__'

    return ssl.__all__


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
    sims = sc.objdict()
    for name, disease in diseases.items():
        sim = ss.Sim(diseases=disease, networks=ss.RandomNet(), **outbreak_pars)
        sim.run()
        sims[name] = sim
        dis = sim.diseases[name]
        res = sim.results[name]
        n_exp = np.array(res.exposed)
        n_infectious = np.array(res.infectious)
        n_inf = np.array(res.infected)
        new = np.array(res.new_infections)
        total = n_exp + n_infectious + np.array(res.susceptible) + np.array(res.recovered)
        assert res.cum_infections[-1] > 0, f'Expected {name} infections'
        assert new.min() >= 0, f'{name} has negative new_infections: seeds are being double-counted'
        assert new[0] == 0, f'{name} counted its seed infections as new infections'
        assert new.sum() == res.cum_infections[-1], f'{name} cumulative infections do not match the sum'
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
    assert yearly.results.cholera.cum_infections[-1] == 0, 'Cholera unexpectedly worked with a yearly timestep'
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

    return s1, s2


@sc.timer()
def test_econ():
    """ Check that ssl.Costs adds up every cost, that ssl.discount() weights by time, and that ssl.icer() compares scenarios with the baseline """
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
    df = ssl.icer(msim, outcome='sis.infection', baseline='None')
    print(df)
    assert df.averted['None'] == 0 and df.averted['Vaccine'] > 0, 'Expected vaccination to avert infections'
    row = df.loc['Vaccine']
    assert np.isclose(row.ICER, (row.cost - df.cost['None'])/row.averted), 'Expected the ICER to be the incremental cost per infection averted'

    return df


if __name__ == '__main__':
    exports  = test_exports()
    diseases = test_diseases()
    hiv      = test_hiv()
    econ     = test_econ()
