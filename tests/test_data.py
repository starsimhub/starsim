"""
Test data by place (ss.data): ages, populations, contact matrices, and administrative units, from starsim_data
"""
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

ssd = pytest.importorskip('starsim_data')
kw = dict(verbose=0)


def make_sir():
    S, I, R = ss.states('S I R')
    return ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3) * I), recovery=ss.Transition(I, R, dur=ss.days(7)), init_prev=0.001, name='sir')


@sc.timer()
def test_contacts():
    sc.heading('Testing contact matrices by name')
    home = ss.contacts('Kenya', layer='home')
    assert home.name == 'home' and home.ref['place'] == 'Kenya' and home.ref['source'] == 'prem2021'
    matrix = ssd.load('Kenya', which='prem2021', kind='all').values
    named = ss.contacts('KE') # By code, as by name
    given = ss.contacts(matrix, ages=list(range(0, 80, 5)))
    sims = [ss.Sim(make_sir(), networks=net, stratify=dict(age=[0, 5, 18, 65]), total_pop=10_000, unit='day', stop=60, **kw).run() for net in [named, given]]
    assert not np.allclose(sims[0].results.sir.I, sims[1].results.sir.I) # A country's matrix brings its people's ages, so has 44% under 18 rather than the world's 30%
    sims[1] = ss.Sim(make_sir(), networks=given, people=ss.people('Kenya'), stratify=dict(age=[0, 5, 18, 65]), total_pop=10_000, unit='day', stop=60, **kw).run()
    assert np.allclose(sims[0].results.sir.I, sims[1].results.sir.I) # The same as the matrix given by hand, with Kenya's people
    assert sims[0].pars.total_pop == 10_000 and sims[0].groups.strats[0].split[:2].sum() > 0.4 # The ages of Kenya, but not its size

    # The data itself is a table labeled by age group, which ss.contacts() takes as it is
    school = ss.data.contacts('Kenya', layer='school')
    assert school.loc['10-14', '10-14'] == school.values[2, 2] and school.index[-1] == '75+' and school.attrs['ref']['layer'] == 'school'
    table = ss.contacts(school)
    assert np.array_equal(table.edges, ss.contacts('Kenya', layer='school').edges) and table.ref == school.attrs['ref'] and table.unit == 'day'
    m = [[10, 3], [3, 5]] # A table not labeled by age group is a matrix over the model's own groups, as a list is
    for df in [pd.DataFrame(m), pd.DataFrame(m, index=['young', 'old'], columns=['young', 'old'])]:
        assert ss.contacts(df).edges is None and np.array_equal(ss.contacts(df).dims['age'], ss.contacts(m).dims['age'])
    assert np.array_equal(ss.contacts(pd.DataFrame(m, index=['0-17', '18+'])).edges, [0, 18])

    # A place a source does not have says which do
    with pytest.raises(KeyError, match='mistry2021'):
        ss.contacts('Japan')
    with pytest.raises(KeyError, match='the place "Kenyaa" was not found'):
        ss.contacts('Kenyaa')
    return sims


@sc.timer()
def test_geo():
    sc.heading('Testing geographies by name')
    kenya = ss.geo.geography('Kenya', level=1)
    nodes = ssd.load_nodes('Kenya', level=1)
    assert len(kenya) == 47 and kenya.names == list(nodes.name)
    assert np.all(kenya.pop > 0) and abs(kenya.pop.sum()/nodes.population.sum() - 1) < 1e-9
    assert np.allclose(kenya.distances, ssd.distances('Kenya', level=1).values, rtol=1e-3) # Great-circle distances in km, as the provider's
    i, j = kenya.names.index('Nairobi'), kenya.names.index('Kiambu')
    assert kenya.adjacency[i, j] and kenya.adjacency[j, i] and not kenya.adjacency[i, i] # Neighbors
    assert not kenya.adjacency[i, kenya.names.index('Mombasa')]

    # A subset, and several countries
    two = ss.geo.geography('Kenya', level=1, places=['Nairobi', 'Kiambu'])
    assert two.names == ['Kiambu', 'Nairobi'] and two.adjacency[0, 1]
    both = ss.geo.geography(['Kenya', 'Uganda'], level=1)
    assert len(both) == 47 + len(ssd.load_nodes('Uganda', level=1)) and both.names[0].startswith('Kenya:')
    with pytest.raises(KeyError, match='the unit "Atlantis" was not found'):
        ss.geo.geography('Kenya', places=['Atlantis'])

    # As the places of a model, with travel between them
    sir = make_sir()
    sir.stratify(patch=kenya, mobility=ss.geo.gravity(away=0.1))
    sim = ss.Sim(sir, total_pop=50e6, unit='day', stop=60, **kw).run()
    shares = sim.results.sir.S.strata[0]/sim.results.sir.S.strata[0].sum()
    assert np.allclose(shares, kenya.pop/kenya.pop.sum()) # People start in each county in proportion to its population
    assert 'starsim_data' in sim.describe(output=True)

    # Gravity gives the shape of the movement and away= its amount (10% if not given); a k that sends people away more than all their time warns
    home = lambda mob: ss.Sim(make_sir().stratify(patch=kenya, mobility=mob), unit='day', stop=60, **kw).init().mixing.time_share.diagonal()
    assert np.allclose(home(ss.geo.gravity()), 0.9) and np.allclose(home(ss.geo.gravity(away=0.2)), 0.8)
    assert ss.Sim(make_sir().stratify(patch=kenya), unit='day', stop=60, **kw).init().pars.total_pop == kenya.pop.sum() # Places with populations give the population size
    with pytest.warns(RuntimeWarning, match='capped'):
        assert np.allclose(home(ss.geo.gravity(k=0.5)), 0) # Everyone away all the time, as the spec's example had it
    with np.errstate(divide='ignore'): # As LASER scales its gravity: the share away is 5% on average, and at most 7%, given as the matrix
        G = kenya.pop[None, :]**0.25/kenya.distances**1.56
    np.fill_diagonal(G, 0)
    G *= 0.05/G.sum(axis=1).mean()
    capped = G*np.minimum(1, 0.07/G.sum(axis=1))[:, None]
    assert np.allclose(1 - home(ss.geo.Mobility(matrix=capped)), capped.sum(axis=1)) and G.sum(axis=1).max() > 0.07 >= capped.sum(axis=1).max() - 1e-12

    # The places say where they came from
    ref = sim.diseases.sir.graph.strats[0].geo.ref
    assert ref['place'] == 'Kenya' and ref['level'] == 1
    return sim


@sc.timer()
def test_people():
    sc.heading('Testing populations by name')
    ppl = ss.people('Kenya', year=2025)
    pop = ss.data.population('Kenya', year=2020) # A table of women and men by age group
    later = ss.data.population('Kenya', year=2030)
    assert np.isclose(ppl.named.total, 0.5*pop.values.sum() + 0.5*later.values.sum()) # Between the decadal values, interpolated
    assert np.array_equal(ss.data.age('Kenya', year=2020)[:-1], np.column_stack([np.arange(0, 101, 5), pop.female + pop.male])) # The age_data of ss.People, ending with 0 people at 105
    assert list(pop.columns) == ['female', 'male'] and pop.index[1] == '5-9' and pop.attrs['ref']['year'] == 2020

    # The size and the age split of a compartmental model, with no agents
    stratify = dict(age=[0, 5, 18, 65])
    ode = ss.Sim(make_sir(), people=ss.people('Kenya', year=2024), stratify=stratify, unit='day', stop=60, **kw).run()
    total = ss.people('Kenya', year=2024).named.total
    assert ode.solver == 'ode' and ode.pars.total_pop == total
    n = ode.results.sir
    initial = n.S.strata[:, 0] + n.I.strata[:, 0] + n.R.strata[:, 0]
    pop = ss.data.population('Kenya', year=2024)
    assert np.isclose(initial[0], pop.loc['0-4'].sum()) # The 0-4 group, which is one WPP age group
    female, male = pop.female.values, pop.male.values

    # As agents, the same ages and the share who are female
    abm = ss.Sim(make_sir(), people=ss.people('Kenya', year=2024, n_agents=20_000), stratify=stratify, unit='day', stop=60, **kw).run()
    assert abm.solver == 'abm' and abm.pars.pop_scale == total/20_000
    share = np.histogram(abm.people.age.values, [0, 5, 18, 65, 200])[0]/20_000
    assert np.allclose(share, abm.groups.strats[0].split, atol=0.01)
    assert abs(abm.people.female.mean() - female.sum()/(female + male).sum()) < 0.01

    # In a compartmental run, people age out of each age group as the population's own pyramid has them do, as agents do
    growth = sc.obj()
    for solver,n_agents in [('ode', None), ('abm', 20_000)]:
        sim = ss.Sim(make_sir(), people=ss.people('Kenya', year=2024, n_agents=n_agents), stratify=dict(age=[0, 15, 65]), solver=solver, use_aging=True, unit='day', stop=365, **kw).run()
        n = sum(sim.results.sir[k].by('age').values for k in 'SIR')
        growth[solver] = n[-1, 2]/n[0, 2] - 1
    assert growth.abm > 0.05 and np.isclose(growth.ode, growth.abm, rtol=0.15) # The 65+ group grows by about 12% in a year; 40% with ages uniform within each group

    # With no named population, the ages of a declared model are the world's, as a guess, stratified or not
    for stratify in [None, dict(age=[0, 15, 65])]:
        ages = ss.Sim(make_sir(), n_agents=20_000, stratify=stratify, unit='day', stop=10, **kw).init().people.age.values
        assert 0.08 < (ages >= 65).mean() < 0.12 and 0.23 < (ages < 15).mean() < 0.27 # Not uniform from 0 to 60 (no one over 65) or to 100 (35%)

    # The year the sim starts, if it has dates, and the people say where they came from
    sim = ss.Sim(make_sir(), people=ss.people('Kenya'), start='1990-01-01', stop='1990-03-01', unit='day', **kw)
    sim.init()
    assert sim.named_people.ref['year'] == 1990
    with pytest.raises(KeyError, match='the place "Kenyaa" was not found'):
        ss.people('Kenyaa')
    return ode, abm


@pytest.fixture
def us_states(tmp_path, monkeypatch):
    """ The US states' datasets of starsim_data, from the files of a copy of its repository if it is installed from one (so offline), else downloaded """
    stores = sc.path(ssd.__file__).parent.parent/'fetched'
    names = ['population_us', 'mistry2021_us']
    if all((stores/ssd.registry.dataset(name).source().store).is_file() for name in names):
        monkeypatch.setattr(ssd.options, 'cache_dir', str(tmp_path))
        for name in names: # As if downloaded: the file, and the record of its digest that fetch() checks
            source = ssd.registry.dataset(name).source()
            path = ssd.cache_dir('stores', create=True)/source.store
            path.write_bytes((stores/source.store).read_bytes())
            ssd.cache._write_sidecar(path, source.url)
    try:
        ssd.fetch('United States:Massachusetts', which='population_us', verbose=False)
    except OSError as E:
        pytest.skip(f'The US states\' data of starsim_data are not available: {E}')
    return


@sc.timer()
def test_us_states(us_states):
    sc.heading('Testing the people and contacts of a US state by name')
    state = 'United States:Massachusetts'
    ppl = ss.people(state)
    assert 6.5e6 < ppl.named.total < 7.5e6 and ppl.named.ref['year'] == 2023 # The US Census for 2023
    assert list(ppl.named.ages) == list(range(0, 85, 5)) # Grouped by 5 years, as a country's
    school = ss.contacts(state, layer='school')
    assert school.ref['source'] == 'mistry2021_us' and len(school.edges) == 85 # By single year of age, from its own study
    sim = ss.Sim(make_sir(), people=ss.people(state), networks=school, stratify=dict(age=[0, 5, 18, 65]), unit='day', stop=30, **kw).run()
    assert sim.pars.total_pop == ppl.named.total and sim.results.sir.I[-1] > sim.results.sir.I[0]
    with pytest.raises(KeyError, match='the place "Kenya:Nairobi" was not found'):
        ss.people('Kenya:Nairobi') # Other countries have no data by unit
    return sim


@sc.timer()
def test_fertility():
    sc.heading('Testing age-specific fertility rates by name')
    if 'fertility' not in ssd.list_datasets():
        pytest.skip(f'starsim_data {ssd.__version__} has no age-specific fertility rates (added in 2.1.0)')
    sim = ss.Sim(people=ss.people('Nigeria', n_agents=2000), demographics=ss.Pregnancy(fertility_rate='Nigeria', burnin=False), start=2000, stop=2003, dt=ss.months(1), **kw).run()
    preg = sim.demographics.pregnancy
    asfr = ssd.load('Nigeria', which='fertility')
    assert np.isclose(preg.fertility_rate_data.loc[2000, 25], asfr.set_index(['Time', 'AgeGrpStart']).ASFR[2000, 25]) # The UN's, by 5-year age group, per 1,000 women
    assert 4 < sim.results.pregnancy.tfr[-12:].mean() < 7 # Nigeria's TFR was about 6
    return sim


if __name__ == '__main__':
    T = sc.timer()
    test_contacts()
    test_geo()
    test_people()
    test_fertility()
    T.toc()
