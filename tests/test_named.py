"""
Test named data: places and contact matrices looked up by name, from starsim_data
"""
import pytest
import numpy as np
import sciris as sc
import starsim as ss

ssd = pytest.importorskip('starsim_data')
kw = dict(verbose=0)


def make_sir():
    S, I, R = ss.states('S I R')
    return ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3) * I), recovery=ss.Transition(I, R, dur=ss.days(7)), init_prev=0.001, name='sir')


def rebuild(code):
    """ The sim that code written by sim.to_code() builds """
    ns = {}
    exec(code, ns)
    return ns['sim']


@sc.timer()
def test_contacts():
    sc.heading('Testing contact matrices by name')
    home = ss.contacts('Kenya', layer='home')
    assert home.name == 'home' and home.ref['place'] == 'Kenya' and home.ref['source'] == 'prem2021'
    matrix = ssd.load('Kenya', which='prem2021', kind='all').values
    named = ss.contacts('KE') # By code, as by name
    given = ss.contacts(matrix, bands=list(range(0, 80, 5)))
    sims = [ss.Sim(make_sir(), networks=net, stratify=dict(age=[0, 5, 18, 65]), total_pop=10_000, unit='day', stop=60, **kw).run() for net in [named, given]]
    assert not np.allclose(sims[0].results.sir.n_I, sims[1].results.sir.n_I) # A country's matrix brings its people's ages, so has 44% under 18 rather than the world's 30%
    sims[1] = ss.Sim(make_sir(), networks=given, people=ss.people('Kenya'), stratify=dict(age=[0, 5, 18, 65]), total_pop=10_000, unit='day', stop=60, **kw).run()
    assert np.allclose(sims[0].results.sir.n_I, sims[1].results.sir.n_I) # The same as the matrix given by hand, with Kenya's people
    assert sims[0].pars.total_pop == 10_000 and sims[0].groups.strats[0].split[:2].sum() > 0.4 # The ages of Kenya, but not its size

    # A place a source does not have says which do
    with pytest.raises(ss.errors.UnknownLocation, match='mistry2021'):
        ss.contacts('Japan')
    with pytest.raises(ss.errors.UnknownLocation):
        ss.contacts('Kenyaa')

    # The spec says where the matrices came from, so the sim can be rebuilt, as data or code
    sim = ss.Sim(make_sir(), networks=[home, ss.contacts('Kenya', layer='school')], stratify=dict(age=[0, 5, 18, 65]), unit='day', stop=60, **kw)
    spec = sim.to_spec()
    assert spec['networks'][1]['data']['layer'] == 'school'
    code = ss.to_code(spec)
    assert sim.hash() == rebuild(code).hash()
    assert "ss.contacts('Kenya', layer='school')" in code
    return sims


@sc.timer()
def test_geo():
    sc.heading('Testing geographies by name')
    kenya = ss.geo.location('Kenya', level=1)
    nodes = ssd.load_nodes('Kenya', level=1)
    assert len(kenya) == 47 and kenya.names == list(nodes.name)
    assert np.all(kenya.pop > 0) and abs(kenya.pop.sum()/nodes.population.sum() - 1) < 1e-9
    assert np.allclose(kenya.distances, ssd.distances('Kenya', level=1).values, rtol=1e-3) # Great-circle distances in km, as the provider's
    i, j = kenya.names.index('Nairobi'), kenya.names.index('Kiambu')
    assert kenya.adjacency[i, j] and kenya.adjacency[j, i] and not kenya.adjacency[i, i] # Neighbours
    assert not kenya.adjacency[i, kenya.names.index('Mombasa')]

    # A subset, and several countries
    two = ss.geo.location('Kenya', level=1, places=['Nairobi', 'Kiambu'])
    assert two.names == ['Kiambu', 'Nairobi'] and two.adjacency[0, 1]
    both = ss.geo.location(['Kenya', 'Uganda'], level=1)
    assert len(both) == 47 + len(ssd.load_nodes('Uganda', level=1)) and both.names[0].startswith('Kenya:')
    with pytest.raises(ss.errors.UnknownLocation):
        ss.geo.location('Kenya', places=['Atlantis'])

    # As the places of a model, with travel between them
    sir = make_sir()
    sir.stratify(patch=kenya, mobility=ss.geo.gravity(away=0.1))
    sim = ss.Sim(sir, total_pop=50e6, unit='day', stop=60, **kw).run()
    shares = sim.results.sir.n_S.strata[0]/sim.results.sir.n_S.strata[0].sum()
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

    # The spec says where the places came from
    spec = sim.to_spec()
    strat = spec['modules'][0]['stratifications'][0]
    assert strat['geo']['place'] == 'Kenya' and strat['geo']['level'] == 1
    sir = make_sir()
    sir.stratify(patch=ss.geo.location('Kenya', level=1))
    code = ss.Sim(sir, unit='day', stop=60, **kw).to_code()
    assert "patch=ss.geo.location('Kenya', level=1)" in code
    return sim


@sc.timer()
def test_people():
    sc.heading('Testing populations by name')
    ppl = ss.people('Kenya', year=2025)
    ages, female, male, _ = ss.named.population('Kenya', year=2020)
    later = ss.named.population('Kenya', year=2030)
    assert np.isclose(ppl.named.total, 0.5*(female + male).sum() + 0.5*(later[1] + later[2]).sum()) # Between the decadal values, interpolated

    # The size and the age split of a compartmental model, with no agents
    stratify = dict(age=[0, 5, 18, 65])
    ode = ss.Sim(make_sir(), people=ss.people('Kenya', year=2024), stratify=stratify, unit='day', stop=60, **kw).run()
    total = ss.people('Kenya', year=2024).named.total
    assert ode.solver == 'ode' and ode.pars.total_pop == total
    n = ode.results.sir
    initial = n.n_S.strata[:, 0] + n.n_I.strata[:, 0] + n.n_R.strata[:, 0]
    _, female, male, _ = ss.named.population('Kenya', year=2024)
    assert np.isclose(initial[0], (female + male)[0]) # The 0-4 group, which is one WPP age group

    # As agents, the same ages and the share who are female
    abm = ss.Sim(make_sir(), people=ss.people('Kenya', year=2024, n_agents=20_000), stratify=stratify, unit='day', stop=60, **kw).run()
    assert abm.solver == 'abm' and abm.pars.pop_scale == total/20_000
    share = np.histogram(abm.people.age.values, [0, 5, 18, 65, 200])[0]/20_000
    assert np.allclose(share, abm.groups.strats[0].split, atol=0.01)
    assert abs(abm.people.female.mean() - female.sum()/(female + male).sum()) < 0.01

    # In a compartmental run, people age out of each age group as the population's own pyramid has them do, as agents do
    growth = sc.objdict()
    for solver,n_agents in [('ode', None), ('abm', 20_000)]:
        sim = ss.Sim(make_sir(), people=ss.people('Kenya', year=2024, n_agents=n_agents), stratify=dict(age=[0, 15, 65]), solver=solver, use_aging=True, unit='day', stop=365, **kw).run()
        n = sum(sim.results.sir[f'n_{k}'].by('age').values for k in 'SIR')
        growth[solver] = n[-1, 2]/n[0, 2] - 1
    assert growth.abm > 0.05 and np.isclose(growth.ode, growth.abm, rtol=0.15) # The 65+ group grows by about 12% in a year; 40% with ages uniform within each group

    # With no named population, the ages of a declared model are the world's, as a guess, stratified or not
    for stratify in [None, dict(age=[0, 15, 65])]:
        ages = ss.Sim(make_sir(), n_agents=20_000, stratify=stratify, unit='day', stop=10, **kw).init().people.age.values
        assert 0.08 < (ages >= 65).mean() < 0.12 and 0.23 < (ages < 15).mean() < 0.27 # Not uniform from 0 to 60 (no one over 65) or to 100 (35%)

    # The year the sim starts, if it has dates; and the spec says where the people came from
    sim = ss.Sim(make_sir(), people=ss.people('Kenya'), start='1990-01-01', stop='1990-03-01', unit='day', **kw)
    sim.init()
    assert sim.to_spec()['config']['people']['year'] == 1990
    assert sim.hash() == rebuild(sim.to_code()).hash()
    assert "people=ss.people('Kenya', year=1990" in sim.to_code()
    with pytest.raises(ss.errors.UnknownLocation):
        ss.people('Kenyaa')
    return ode, abm


if __name__ == '__main__':
    T = sc.timer()
    test_contacts()
    test_geo()
    test_people()
    T.toc()
