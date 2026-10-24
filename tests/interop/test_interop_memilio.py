"""
Test importing MEmilio models
"""
import subprocess
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss
import starsim.interop
import paths
from starsim.interop.memilio import load_memilio

lfd = paths.lfd # The examples, with the originals' results, if this folder is here
python = lfd / '_envs/memilio/bin/python' # MEmilio's environment
dump = sc.path(ss.__file__).parent / 'interop/dump_memilio.py'
kw = dict(verbose=False)

graph_script = '''
""" Three places in two age groups, run as a MobilitySimulation (or, with STUDY=n, a GraphParameterStudy of n runs); writes MEmilio's results next to it """
import os
import numpy as np
import memilio.simulation as mio
import memilio.simulation.osecir as osecir

mio.set_log_level(mio.LogLevel.Off)
G = 2
study = int(os.environ.get('STUDY', 0))
pops = [[60_000, 40_000], [30_000, 20_000], [15_000, 10_000]]
M = np.array([[0, 0.08, 0.04], [0.1, 0, 0.05], [0.06, 0.06, 0]]) # The share of each place's people who commute to each other place

def model(i):
    m = osecir.Model(G)
    p = m.parameters
    for a in range(G):
        A = mio.AgeGroup(a)
        p.TimeExposed[A], p.TimeInfectedNoSymptoms[A], p.TimeInfectedSymptoms[A], p.TimeInfectedSevere[A], p.TimeInfectedCritical[A] = 3.2, 2.0, [5.8, 6.5][a], 9.5, 7.1
        p.TransmissionProbabilityOnContact[A], p.RelativeTransmissionNoSymptoms[A], p.RecoveredPerInfectedNoSymptoms[A] = [0.05, 0.08][a], 0.7, [0.3, 0.1][a]
        p.RiskOfInfectionFromSymptomatic[A], p.MaxRiskOfInfectionFromSymptomatic[A] = 0.25, 0.25
        p.SeverePerInfectedSymptoms[A], p.CriticalPerSevere[A], p.DeathsPerCritical[A] = [0.1, 0.3][a], 0.25, 0.3
        if study:
            p.TransmissionProbabilityOnContact[A].set_distribution(mio.ParameterDistributionUniform([0.04, 0.06][a], [0.06, 0.10][a]))
            p.TimeExposed[A].set_distribution(mio.ParameterDistributionUniform(2.5, 4.0))
        m.populations[A, osecir.InfectionState.Exposed] = 50 if i == 0 else 0
        m.populations.set_difference_from_group_total_AgeGroup((A, osecir.InfectionState.Susceptible), pops[i][a])
    cm = mio.ContactMatrix(np.array([[10.0, 3.0], [3.0, 8.0]]))
    cm.add_damping(mio.Damping(np.full((G, G), 0.4), t=30.0))
    p.ContactPatterns.cont_freq_mat[0] = cm
    if study:
        value = mio.UncertainValue(0.4)
        value.set_distribution(mio.ParameterDistributionUniform(0.2, 0.6))
        p.ContactPatterns.dampings = [mio.DampingSampling(value=value, level=0, type=0, time=30.0, matrix_indices=[0], group_weights=np.ones(G))]
    m.apply_constraints()
    return m

def coefficients(i, j):
    c = np.full(10*G, M[i, j])
    c[[a*10 + int(osecir.InfectionState.Dead) for a in range(G)]] = 0
    return c

def symptomatic(node):
    r = osecir.interpolate_simulation_result(node.property.result).as_ndarray()[1:]
    return sum(r[a*10 + 4] + r[a*10 + 5] for a in range(G))

if study:
    graph = osecir.ModelGraph()
    for i in range(3):
        graph.add_node(i, model(i))
    for i, j in zip(*np.nonzero(M)):
        graph.add_edge(int(i), int(j), coefficients(i, j))
    runs = []
    def handle(g, run):
        p = g.get_node(0).property.model.parameters
        draws = [p.TransmissionProbabilityOnContact[mio.AgeGroup(a)].value for a in range(G)] + [p.TimeExposed[mio.AgeGroup(0)].value, p.ContactPatterns.dampings[0].value.value]
        runs.append(draws + [sum(symptomatic(g.get_node(i)) for i in range(3)).max()])
    osecir.GraphParameterStudy(graph, 0, 60, 0.5, study).run(handle)
    np.save('study.npy', np.array(runs))
else:
    graph = osecir.MobilityGraph()
    for i in range(3):
        graph.add_node(id=i, model=model(i), t0=0)
    for i, j in zip(*np.nonzero(M)):
        graph.add_edge(int(i), int(j), mio.MobilityParameters(coefficients(i, j)))
    sim = osecir.MobilitySimulation(graph, 0, dt=0.5)
    sim.advance(60)
    np.save('graph.npy', np.array([symptomatic(sim.graph.get_node(i))[:61] for i in range(3)]))
'''


def seir_model():
    """ An SEIR in two age groups, with a damping on day 10, as dump_memilio.py writes it """
    base = [[4.0, 1.0], [1.0, 3.0]]
    return dict(
        format = 'memilio',
        model = 'oseir',
        groups = 2,
        states = ['Susceptible', 'Exposed', 'Infected', 'Recovered'],
        parameters = dict(
            TransmissionProbabilityOnContact = dict(kind='age', value=[0.05, 0.08]),
            TimeExposed = dict(kind='age', value=[3.0, 4.0]),
            TimeInfected = dict(kind='age', value=[6.0, 6.0]),
        ),
        other = {},
        populations = dict(Susceptible=[5985.0, 3990.0], Exposed=[10.0, 0.0], Infected=[5.0, 10.0], Recovered=[0.0, 0.0]),
        contacts = dict(
            matrices = [dict(baseline=base, minimum=[[0, 0], [0, 0]], dampings=[dict(coeffs=[[0.5, 0.5], [0.5, 0.5]], time=10.0, level=0, type=0)])],
            samples = [dict(time=0.0, matrix=base), dict(time=9.0, matrix=base), dict(time=10.0, matrix=(np.array(base)/2).tolist())],
        ),
        run = dict(t0=0.0, tmax=60.0, dt=0.1, function='simulate'),
    )


def test_seir():
    sc.heading('Testing importing a MEmilio SEIR by age')
    sim = load_memilio(seir_model(), **kw)
    assert not sim.imported.incomplete
    sim.run()
    res = sim.results.oseir
    assert res.n_Infected.select(age=1).values[0] == 10, 'Each group should start with its initial numbers'

    # The same model by hand
    Susceptible, Exposed, Infected, Recovered = ss.states(4)
    contacts = ss.timeseries({0: 1.0, 9: 1.0, 10: 0.5}, mode='cosine') # MEmilio smooths a damping over the day before it
    seir = ss.Disease(
        infection = Susceptible >> Exposed @ contacts*ss.by([0.05, 0.08], dims=['age'])*Infected,
        onset     = Exposed >> Infected @ ss.by([1/3, 1/4], dims=['age']),
        recovery  = Infected >> Recovered @ 1/6,
        init_n    = dict(Exposed=ss.by([10, 0], dims=['age']), Infected=ss.by([5, 10], dims=['age'])),
        name      = 'oseir',
    )
    seir.stratify(age=['0', '1'], split=[0.6, 0.4])
    ref = ss.Sim(seir, networks=ss.contacts(age=[[4.0, 1.0], [1.0, 3.0]]), total_pop=10_000, unit='day', stop=ss.days(60), solver='ode', verbose=0).run()
    for s in ['Susceptible', 'Exposed', 'Infected', 'Recovered']:
        assert np.allclose(res[f'n_{s}'].values, ref.results.oseir[f'n_{s}'].values, rtol=1e-9), f'{s} should match the model written by hand'
    return sim


def test_secir():
    sc.heading('Testing importing a MEmilio SECIR with test and trace and ICU capacity')
    d = seir_model()
    zeros = [0.0, 0.0]
    d.update(model='osecir', states=['Susceptible', 'Exposed', 'InfectedNoSymptoms', 'InfectedNoSymptomsConfirmed', 'InfectedSymptoms', 'InfectedSymptomsConfirmed', 'InfectedSevere', 'InfectedCritical', 'Recovered', 'Dead'])
    d['populations'] = dict(Susceptible=[5940.0, 3960.0], Exposed=[30.0, 20.0], InfectedNoSymptoms=[20.0, 10.0], InfectedNoSymptomsConfirmed=zeros, InfectedSymptoms=[10.0, 10.0],
                            InfectedSymptomsConfirmed=zeros, InfectedSevere=zeros, InfectedCritical=zeros, Recovered=zeros, Dead=zeros)
    age = lambda *v: dict(kind='age', value=list(v))
    num = lambda v: dict(kind='number', value=v)
    d['parameters'] = dict(
        TransmissionProbabilityOnContact=age(0.1, 0.15), TimeExposed=age(3.2, 3.2), TimeInfectedNoSymptoms=age(2.0, 2.0), TimeInfectedSymptoms=age(5.8, 6.0),
        TimeInfectedSevere=age(9.5, 9.5), TimeInfectedCritical=age(7.1, 7.1), RelativeTransmissionNoSymptoms=age(0.7, 0.7), RecoveredPerInfectedNoSymptoms=age(0.3, 0.1),
        RiskOfInfectionFromSymptomatic=age(0.25, 0.25), MaxRiskOfInfectionFromSymptomatic=age(0.45, 0.45), SeverePerInfectedSymptoms=age(0.2, 0.3),
        CriticalPerSevere=age(0.25, 0.4), DeathsPerSevere=age(0.0, 0.05), DeathsPerCritical=age(0.3, 0.5),
        Seasonality=num(0.2), StartDay=num(60.0), TestAndTraceCapacity=num(20.0), TestAndTraceCapacityMaxRisk=num(5.0), ICUCapacity=num(15.0),
    )
    with pytest.warns(ss.errors.ImportIncomplete): # The share who recover without symptoms differs by age where the groups are summed, for test and trace
        sim = load_memilio(d, **kw)
    assert [k for k,_ in sim.imported.incomplete] == ['RecoveredPerInfectedNoSymptoms by age group']
    sim.run()
    dis = sim.diseases[0]
    assert list(dis.transitions.keys()) == ['infection', 'onset', 'symptoms', 'severe', 'critical', 'severe_recovery', 'severe_death', 'critical_recovery', 'death']
    res = sim.results.osecir
    assert 'n_InfectedNoSymptomsConfirmed' not in res and 'n_Dead' not in res, 'Empty confirmed states are left out, and the dead leave'
    alive = sum(res[f'n_{s}'].values for s in ['Susceptible', 'Exposed', 'InfectedNoSymptoms', 'InfectedSymptoms', 'InfectedSevere', 'InfectedCritical', 'Recovered'])
    dead = res.death.cum.values + res.severe_death.cum.values
    assert np.allclose(alive + dead, 10_000), 'Everyone is alive or has died'
    assert res.n_InfectedCritical.values.max() < 15, 'The ICU should not be overfilled'
    ns = {}
    exec(sim.to_code(), ns)
    assert ns['sim'].hash() == sim.hash(), 'The code should build the same model'
    return sim


def test_graph_json():
    sc.heading('Testing importing a MEmilio graph of two places, as dump_memilio.py writes it')
    d = seir_model()
    pops = [d['populations'], dict(Susceptible=[2000.0, 1000.0], Exposed=[0.0, 0.0], Infected=[0.0, 0.0], Recovered=[0.0, 0.0])]
    coeffs = lambda c: [c]*8 # Each age group and state, as MEmilio's flat index
    d['graph'] = dict(nodes=[dict(id=i, parameters=d['parameters'], other={}, populations=pop, population_distributions=None, contacts=d['contacts']) for i,pop in enumerate(pops)],
                      edges=[dict(start=0, end=1, coefficients=coeffs(0.1), dampings=0), dict(start=1, end=0, coefficients=coeffs(0.2), dampings=0)])
    sim = load_memilio(d, **kw)
    assert not sim.imported.incomplete
    res = sim.run().results.oseir
    assert np.isclose(res.n_Susceptible.select(age=0, patch='1').values[0], 2000), 'Each place and group should start with its own numbers'
    assert res.n_Recovered.select(patch='1').values[-1] > 10, 'The second place should be infected by commuters'

    # The same model by hand: commuters are away half the time
    Susceptible, Exposed, Infected, Recovered = ss.states(4)
    contacts = ss.timeseries({0: 1.0, 9: 1.0, 10: 0.5}, mode='cosine')
    seir = ss.Disease(
        infection = Susceptible >> Exposed @ contacts*ss.by([0.05, 0.08], dims=['age'])*Infected,
        onset     = Exposed >> Infected @ ss.by([1/3, 1/4], dims=['age']),
        recovery  = Infected >> Recovered @ 1/6,
        init_n    = dict(Exposed=ss.by([[10, 0], [0, 0]], dims=['age', 'patch']), Infected=ss.by([[5, 0], [10, 0]], dims=['age', 'patch'])),
        name      = 'oseir',
    )
    totals = np.array([[6000, 2000], [4000, 1000]])
    seir.stratify(age=['0', '1'], split=[8/13, 5/13])
    seir.stratify(patch=['0', '1'], split=ss.by(totals/totals.sum(), dims=['age', 'patch']), mobility=ss.geo.Mobility(matrix=[[0, 0.05], [0.1, 0]], kind='commute'))
    ref = ss.Sim(seir, networks=ss.contacts(age=[[4.0, 1.0], [1.0, 3.0]]), total_pop=13_000, unit='day', stop=ss.days(60), solver='ode', verbose=0).run()
    assert np.allclose(res.n_Infected.values, ref.results.oseir.n_Infected.values, rtol=1e-9), 'The import should match the model written by hand'
    return sim


@pytest.mark.skipif(not python.exists(), reason='needs the MEmilio environment')
def test_graph(tmp_path):
    sc.heading('Testing importing a MEmilio MobilityGraph against MEmilio\'s own run')
    (tmp_path/'graph.py').write_text(graph_script)
    subprocess.run([str(python), '-I', str(dump), 'graph.py', 'graph.json'], check=True, cwd=tmp_path, capture_output=True)
    sim = ss.interop.load(str(tmp_path/'graph.json'), format='memilio', **kw)
    assert not sim.imported.incomplete
    res = sim.run().results.osecir.n_InfectedSymptoms
    theirs = np.load(tmp_path/'graph.npy') # The number with symptoms in each place on each day
    total = theirs.sum(0)
    assert np.abs(res.values - total).max() < 0.02*total.max(), 'The total with symptoms should be within 2% of MEmilio\'s peak' # 1.2%
    for i in range(3): # Each place differs more, mostly because MEmilio's commuters with symptoms commute less (see the importer's note)
        ours = res.select(patch=str(i)).values
        assert np.abs(ours - theirs[i]).max() < 0.25*theirs[i].max(), f'Place {i} should be within 25% of MEmilio\'s peak there' # 5%, 10%, and 19%
    return sim


@pytest.mark.skipif(not python.exists(), reason='needs the MEmilio environment')
def test_study(tmp_path):
    sc.heading('Testing importing a MEmilio GraphParameterStudy, with its distributions as priors')
    (tmp_path/'graph.py').write_text(graph_script)
    subprocess.run([str(python), '-I', str(dump), 'graph.py', 'study.json'], check=True, cwd=tmp_path, capture_output=True, env=dict(STUDY='10'))
    sim = ss.interop.load(str(tmp_path/'study.json'), format='memilio', **kw)
    assert not sim.imported.incomplete and sim.imported.runs == 10

    # MEmilio's draws, given to the import: TransmissionProbabilityOnContact is drawn in each age group from its own range, so the import draws a share of the range
    theirs = np.load(tmp_path/'study.npy') # Each run's draws and its peak with symptoms
    lo, hi = np.array([0.04, 0.06]), np.array([0.06, 0.10])
    draw = (theirs[:, :2] - lo)/(hi - lo)
    pars = [{'TransmissionProbabilityOnContact_draw[0]': a, 'TransmissionProbabilityOnContact_draw[1]': b, 'TimeExposed': te, 'damping': damp} for (a,b),te,damp in zip(draw, theirs[:, 2], theirs[:, 3])]
    peaks = np.asarray(ss.parallel(sim, pars=pars).results.osecir.n_InfectedSymptoms.values).max(1)
    assert np.allclose(peaks, theirs[:, 4], rtol=0.03), 'Each run should give MEmilio\'s peak with the same draws' # Within 2.1%

    # Drawn from the priors, as MEmilio's study draws them
    msim = ss.parallel(sim, n_runs=10, priors=True)
    run_pars = msim.run_pars
    assert run_pars['damping'].between(0.2, 0.6).all() and run_pars['TimeExposed'].between(2.5, 4.0).all()
    assert run_pars['TransmissionProbabilityOnContact_draw[1]'].nunique() == 10, 'Each run should draw its own value for each age group'
    return msim


@pytest.mark.skipif(not python.exists(), reason='needs the MEmilio environment')
def test_secirvvs(tmp_path):
    sc.heading('Testing writing MEmilio\'s SECIRVVS example')
    script = python.parent.parent/'src/pycode/examples/simulation/ode_secirvvs_simple.py'
    subprocess.run([str(python), '-I', str(dump), str(script), 'model.json'], check=True, cwd=tmp_path, capture_output=True)
    d = sc.loadjson(tmp_path/'model.json')
    assert d['model'] == 'osecirvvs' and 'InfectiousnessNewVariant' in d['other'], 'A parameter the bindings cannot read should be listed, not crash the dumper'
    with pytest.raises(NotImplementedError): # Its equations are not written in the importer yet
        load_memilio(d, **kw)
    return d


@pytest.mark.skipif(not python.exists(), reason='needs the MEmilio environment')
@pytest.mark.parametrize('example', ['secir', 'secir_ageres'])
def test_examples(example, tmp_path):
    sc.heading(f'Testing importing the MEmilio example {example} against the original')
    folder = lfd / 'examples/memilio' / example
    script = next((folder / 'original').glob('*.py'))
    path = tmp_path / 'model.json'
    subprocess.run([str(python), '-I', str(dump), str(script), str(path)], check=True, cwd=tmp_path, capture_output=True)
    sim = ss.interop.load(str(path), format=ss.interop.sniff(str(path)), **kw).run()
    ref = pd.read_csv(folder / 'reference/results.csv')
    total = ref[ref.time == 0].value.sum()
    res = sim.results.osecir
    for s in ['Susceptible', 'Exposed', 'InfectedNoSymptoms', 'InfectedSymptoms', 'InfectedSevere', 'InfectedCritical', 'Recovered', 'Dead']:
        theirs = ref[ref.series == s].sort_values('time').value.values
        ours = res.death.cum.values if s == 'Dead' else res[f'n_{s}'].values
        diff = np.abs(ours - theirs).max()/total
        assert diff < 1e-3, f'{s} should be within 0.1% of the population of the original, not {diff:.2%}' # The original's own interpolation error is about this
    return sim


if __name__ == '__main__':
    T = sc.timer()
    sim1 = test_seir()
    sim2 = test_secir()
    sim3 = test_graph_json()
    T.toc()
