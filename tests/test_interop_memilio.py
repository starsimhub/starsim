"""
Test importing MEmilio models
"""
import subprocess
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss
from starsim.interop.memilio import load_memilio

lfd = sc.path('/home/cliffk/idm/lingua_franca_design') # The examples, with the originals' results, if this folder is here
python = lfd / '_envs/memilio/bin/python' # MEmilio's environment
dump = sc.path(ss.__file__).parent / 'interop/dump_memilio.py'
kw = dict(verbose=False)


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
        init      = dict(Exposed=ss.by([10, 0], dims=['age']), Infected=ss.by([5, 10], dims=['age'])),
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
    with pytest.warns(RuntimeWarning, match='W1601'): # The share who recover without symptoms differs by age where the groups are summed, for test and trace
        sim = load_memilio(d, **kw)
    assert [k for k,_ in sim.imported.incomplete] == ['RecoveredPerInfectedNoSymptoms by age group']
    sim.run()
    dis = sim.diseases[0]
    assert list(dis.transitions.keys()) == ['infection', 'onset', 'symptoms', 'severe', 'critical', 'severe_recovery', 'severe_death', 'critical_recovery', 'death']
    res = sim.results.osecir
    assert 'n_InfectedNoSymptomsConfirmed' not in res and 'n_Dead' not in res, 'Empty confirmed states are left out, and the dead leave'
    alive = sum(res[f'n_{s}'].values for s in ['Susceptible', 'Exposed', 'InfectedNoSymptoms', 'InfectedSymptoms', 'InfectedSevere', 'InfectedCritical', 'Recovered'])
    dead = res.cum_death.values + res.cum_severe_death.values
    assert np.allclose(alive + dead, 10_000), 'Everyone is alive or has died'
    assert res.n_InfectedCritical.values.max() < 15, 'The ICU should not be overfilled'
    ns = {}
    exec(sim.to_code(), ns)
    assert ns['sim'].hash() == sim.hash(), 'The code should build the same model'
    return sim


@pytest.mark.skipif(not python.exists(), reason='needs the MEmilio environment')
@pytest.mark.parametrize('example', ['secir', 'secir_ageres'])
def test_examples(example, tmp_path):
    sc.heading(f'Testing importing the MEmilio example {example} against the original')
    folder = lfd / 'examples/memilio' / example
    script = next((folder / 'original').glob('*.py'))
    path = tmp_path / 'model.json'
    subprocess.run([str(python), '-I', str(dump), str(script), str(path)], check=True, cwd=tmp_path, capture_output=True)
    sim = ss.load(str(path), format=ss.interop.sniff(str(path)), **kw).run()
    ref = pd.read_csv(folder / 'reference/results.csv')
    total = ref[ref.time == 0].value.sum()
    res = sim.results.osecir
    for s in ['Susceptible', 'Exposed', 'InfectedNoSymptoms', 'InfectedSymptoms', 'InfectedSevere', 'InfectedCritical', 'Recovered', 'Dead']:
        theirs = ref[ref.series == s].sort_values('time').value.values
        ours = res.cum_death.values if s == 'Dead' else res[f'n_{s}'].values
        diff = np.abs(ours - theirs).max()/total
        assert diff < 1e-3, f'{s} should be within 0.1% of the population of the original, not {diff:.2%}' # The original's own interpolation error is about this
    return sim


if __name__ == '__main__':
    T = sc.timer()
    sim1 = test_seir()
    sim2 = test_secir()
    T.toc()
