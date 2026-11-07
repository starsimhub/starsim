"""
Test Starsim's own JSON form of a model: ss.export_to(sim, 'json') and ss.import_from('model.json') give the same results, for the library's models and a stratified one
"""
import json
import tempfile
import warnings
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl

kw = dict(verbose=0)


def two_patch():
    """ A stratified SIR with values by place and bounds, as in camdl's two-patch example """
    S, I, R = ss.states('S I R')
    gamma = ss.par(0.1, bounds=[0.01, 1.0])
    R0 = ss.par(ss.by(patch=dict(urban=2.5, rural=1.8)), bounds=[0.5, 10])
    infection = S >> I @ R0*gamma*I
    recovery = I >> R @ gamma
    sir = ss.Disease(infection, recovery, init_n=dict(I=ss.by(patch=dict(urban=10, rural=1))), name='sir')
    sir.stratify(patch=['urban', 'rural'])
    return ss.Sim(sir, total_pop=2000, unit='day', stop=200, **kw)


def same(sim):
    """ Export a sim, import it, and check the two give the same results """
    text = ss.export_to(sim, 'json', verbose=False)
    sim2 = ss.import_from(json.loads(text), format='json', verbose=0)
    sim.run()
    sim2.run()
    a, b = sim.results.flatten(), sim2.results.flatten()
    assert set(a) == set(b), f'The results differ: {set(a) ^ set(b)}'
    for key in a:
        if np.asarray(a[key]).dtype.kind == 'f':
            assert np.allclose(a[key], b[key], equal_nan=True), f'"{key}" differs after the round trip'
    return text, sim2


@sc.timer()
def test_two_patch(tmp_path):
    sc.heading('Testing the JSON form of a stratified model, through a file')
    sim = two_patch()
    filename = tmp_path/'sir.json'
    ss.export_to(sim, 'json', filename=filename, verbose=False)
    assert ss.converters.sniff(filename) == 'json'
    sim2 = ss.import_from(filename, verbose=0)
    assert 'sir.stratify(patch=' in sim2.imported.code
    model = sc.loadjson(filename)
    d = model['diseases'][0]
    assert d['transitions'] == dict(infection='S >> I @ R0 * gamma * I', recovery='I >> R @ gamma'), 'Each transition should be one line, as in the code'
    assert d['pars']['R0'] == dict(value=dict(by=dict(patch=dict(urban=2.5, rural=1.8))), bounds=[0.5, 10])
    same(two_patch())
    return model


@pytest.mark.parametrize('name', ['COVID', 'Gonorrhea', 'Influenza', 'Malaria', 'MalariaImmunity', 'Pertussis', 'Polio', 'RSV', 'TB', 'Typhoid', 'Measles', 'Cholera'])  # Not Ebola, whose outcomes (burial, deaths) the JSON form does not carry yet
def test_library(name):
    sc.heading(f'Testing the JSON form of ssl.{name}')
    cls = getattr(ssl, name)
    nets = ss.SexualNet() if name == 'Gonorrhea' else ss.RandomNet(n_contacts=5) if name == 'Measles' else None
    sim = ss.Sim(cls(), networks=nets, n_agents=2000 if nets else None, stop=ss.years(1), rand_seed=3, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ss.errors.LossyExport) # Everything in these is carried
        text, sim2 = same(sim)
    return text


@sc.timer()
def test_folded():
    sc.heading('Testing that results used in rates are written by name')
    text = ss.export_to(ss.Sim(ssl.MalariaImmunity(), stop=ss.years(1), **kw), 'json', verbose=False)
    tr = json.loads(text)['diseases'][0]['transitions']
    assert tr['recovery'] == 'y2 >> x1 @ R1' and tr['inoculation'] == 'x1 >> x2 @ perday(1) * h'
    return text


@sc.timer()
def test_lossy():
    sc.heading('Testing what the JSON form does not carry yet')
    sim = ss.Sim(ssl.Syphilis(), networks=ss.SexualNet(), n_agents=1000, stop=ss.years(1), **kw) # Its modifiers are not carried
    with pytest.warns(ss.errors.LossyExport, match='modifier'):
        ss.export_to(sim, 'json')
    return


@sc.timer()
def test_errors():
    sc.heading('Testing that a file runs no code and names what is wrong')
    model = json.loads(ss.export_to(two_patch(), 'json', verbose=False))
    bad = sc.dcp(model)
    bad['diseases'][0]['transitions']['recovery'] = "I >> R @ __import__('os').getcwd()"
    with pytest.raises(ValueError, match='not something a model can say'):
        ss.import_from(bad, format='json')
    bad = sc.dcp(model)
    bad['diseases'][0]['transitions']['recovery'] = 'I >> R @ gama'
    with pytest.raises(ValueError, match='"gama" is not a state, parameter'):
        ss.import_from(bad, format='json')
    bad = sc.dcp(model)
    bad['networks'] = [dict(type='os.system')]
    with pytest.raises(ValueError, match='not a module of Starsim'):
        ss.import_from(bad, format='json')
    return


if __name__ == '__main__':
    T = sc.timer()
    with tempfile.TemporaryDirectory() as tmp:
        test_two_patch(sc.path(tmp))
    for name in ['COVID', 'Malaria', 'MalariaImmunity']:
        test_library(name)
    test_folded()
    test_lossy()
    test_errors()
    T.toc()
