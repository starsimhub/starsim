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
    return ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.3)), recovery=ss.progression('I -> R', dur=ss.days(7)), init_prev=0.001, name='sir')


@sc.timer()
def test_contacts():
    sc.heading('Testing contact matrices by name')
    home = ss.contacts('Kenya', layer='home')
    assert home.name == 'home' and home.ref['place'] == 'Kenya' and home.ref['source'] == 'prem2021'
    matrix = ssd.load('Kenya', which='prem2021', kind='all').values
    named = ss.contacts('KE') # By code, as by name
    given = ss.contacts(matrix, age=list(range(0, 80, 5)))
    sims = [ss.Sim(make_sir(), networks=net, stratify=dict(age=[0, 5, 18, 65]), unit='day', stop=60, **kw).run() for net in [named, given]]
    assert np.allclose(sims[0].results.sir.n_I, sims[1].results.sir.n_I) # The same as the matrix given by hand

    # A place a source does not have says which do
    with pytest.raises(ss.CodedError, match='mistry2021'):
        ss.contacts('Japan')
    with pytest.raises(ss.CodedError, match='E605'):
        ss.contacts('Kenyaa')

    # The spec says where the matrices came from, so the sim can be rebuilt, as data or code
    sim = ss.Sim(make_sir(), networks=[home, ss.contacts('Kenya', layer='school')], stratify=dict(age=[0, 5, 18, 65]), unit='day', stop=60, **kw)
    spec = sim.to_spec()
    assert spec['networks'][1]['data']['layer'] == 'school'
    assert sim.hash() == ss.from_spec(spec).hash()
    code = ss.to_code(spec)
    assert "ss.contacts('Kenya', layer='school')" in code
    return sims


@sc.timer()
def test_geo():
    sc.heading('Testing geographies by name')
    kenya = ss.geo('Kenya', level=1)
    nodes = ssd.load_nodes('Kenya', level=1)
    assert len(kenya) == 47 and kenya.names == list(nodes.name)
    assert np.all(kenya.pop > 0) and abs(kenya.pop.sum()/nodes.population.sum() - 1) < 1e-9
    assert np.allclose(kenya.distances, ssd.distances('Kenya', level=1).values, rtol=1e-3) # Great-circle distances in km, as the provider's
    i, j = kenya.names.index('Nairobi'), kenya.names.index('Kiambu')
    assert kenya.adjacency[i, j] and kenya.adjacency[j, i] and not kenya.adjacency[i, i] # Neighbours
    assert not kenya.adjacency[i, kenya.names.index('Mombasa')]

    # A subset, and several countries
    two = ss.geo('Kenya', level=1, places=['Nairobi', 'Kiambu'])
    assert two.names == ['Kiambu', 'Nairobi'] and two.adjacency[0, 1]
    both = ss.geo(['Kenya', 'Uganda'], level=1)
    assert len(both) == 47 + len(ssd.load_nodes('Uganda', level=1)) and both.names[0].startswith('Kenya:')
    with pytest.raises(ss.CodedError, match='E605'):
        ss.geo('Kenya', places=['Atlantis'])

    # As the places of a model, with travel between them
    sir = make_sir()
    sir.stratify(patch=kenya, mobility=ss.gravity(k=1e-4))
    sim = ss.Sim(sir, total_pop=50e6, unit='day', stop=60, **kw).run()
    shares = sim.results.sir.n_S.strata[0]/sim.results.sir.n_S.strata[0].sum()
    assert np.allclose(shares, kenya.pop/kenya.pop.sum()) # People start in each county in proportion to its population
    assert 'starsim_data' in sim.describe(output=True)

    # The spec says where the places came from
    spec = sim.to_spec()
    strat = spec['modules'][0]['stratifications'][0]
    assert strat['geo']['place'] == 'Kenya' and strat['geo']['level'] == 1
    sir = make_sir()
    sir.stratify(patch=ss.geo('Kenya', level=1))
    code = ss.Sim(sir, unit='day', stop=60, **kw).to_code()
    assert "patch=ss.geo('Kenya', level=1)" in code
    return sim


if __name__ == '__main__':
    T = sc.timer()
    test_contacts()
    test_geo()
    T.toc()
