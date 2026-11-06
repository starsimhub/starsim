"""
Test importing MEmilio models, and exporting to MEmilio

The exports are run in MEmilio's environment, if `STARSIM_LF` gives the lingua_franca repo.
"""
import os
import subprocess
import numpy as np
import pytest
import sciris as sc
import starsim as ss
from starsim.converters.memilio import load_memilio, to_memilio

kw = dict(verbose=False)
lf = os.environ.get('STARSIM_LF')
memilio_python = os.path.join(lf, '_envs', 'memilio', 'bin', 'python') if lf else None
has_memilio = memilio_python is not None and os.path.exists(memilio_python)

def seir_model():
    """ An SEIR in two age groups, with a damping on day 10, as memilio.py writes it """
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
    assert res.Infected.select(age=1).values[0] == 10, 'Each group should start with its initial numbers'

    # The same model by hand
    Susceptible, Exposed, Infected, Recovered = ss.states(4)
    contacts = ss.timeseries({0: 1.0, 9: 1.0, 10: 0.5}, interpolation='cosine') # MEmilio smooths a damping over the day before it
    seir = ss.Disease(
        infection = Susceptible >> Exposed @ contacts*ss.by([0.05, 0.08], dims=['age'])*Infected,
        onset     = Exposed >> Infected @ ss.by([1/3, 1/4], dims=['age']),
        recovery  = Infected >> Recovered @ 1/6,
        init_n    = dict(Exposed=ss.by([10, 0], dims=['age']), Infected=ss.by([5, 10], dims=['age'])),
        name      = 'oseir',
    )
    seir.stratify(age=['0', '1'], fractions=[0.6, 0.4])
    ref = ss.Sim(seir, networks=ss.contacts([[4.0, 1.0], [1.0, 3.0]]), total_pop=10_000, unit='day', stop=ss.days(60), solver='ode', verbose=0).run()
    for s in ['Susceptible', 'Exposed', 'Infected', 'Recovered']:
        assert np.allclose(res[s].values, ref.results.oseir[s].values, rtol=1e-9), f'{s} should match the model written by hand'
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
    assert 'InfectedNoSymptomsConfirmed' not in res and 'Dead' not in res, 'Empty confirmed states are left out, and the dead leave'
    alive = sum(res[s].values for s in ['Susceptible', 'Exposed', 'InfectedNoSymptoms', 'InfectedSymptoms', 'InfectedSevere', 'InfectedCritical', 'Recovered'])
    dead = sum(res[f].cum.values - res[f].values for f in ['death', 'severe_death']) # Deaths before each time point (the cumulative total includes the step from it)
    assert np.allclose(alive + dead, 10_000), 'Everyone is alive or has died'
    assert res.InfectedCritical.values.max() < 15, 'The ICU should not be overfilled'
    ns = {}
    exec(sim.imported.code, ns) # The code stands alone, and gives the same results
    ns['sim'].pars.verbose = 0
    assert np.array_equal(ns['sim'].run().results.osecir.Susceptible, res.Susceptible), 'The code should build the same model'
    return sim


def test_graph_json():
    sc.heading('Testing importing a MEmilio graph of two places, as memilio.py writes it')
    d = seir_model()
    pops = [d['populations'], dict(Susceptible=[2000.0, 1000.0], Exposed=[0.0, 0.0], Infected=[0.0, 0.0], Recovered=[0.0, 0.0])]
    coeffs = lambda c: [c]*8 # Each age group and state, as MEmilio's flat index
    d['graph'] = dict(nodes=[dict(id=i, parameters=d['parameters'], other={}, populations=pop, population_distributions=None, contacts=d['contacts']) for i,pop in enumerate(pops)],
                      edges=[dict(start=0, end=1, coefficients=coeffs(0.1), dampings=0), dict(start=1, end=0, coefficients=coeffs(0.2), dampings=0)])
    sim = load_memilio(d, **kw)
    assert not sim.imported.incomplete
    res = sim.run().results.oseir
    assert np.isclose(res.Susceptible.select(age=0, patch='1').values[0], 2000), 'Each place and group should start with its own numbers'
    assert res.Recovered.select(patch='1').values[-1] > 10, 'The second place should be infected by commuters'

    # The same model by hand: commuters are away half the time
    Susceptible, Exposed, Infected, Recovered = ss.states(4)
    contacts = ss.timeseries({0: 1.0, 9: 1.0, 10: 0.5}, interpolation='cosine')
    seir = ss.Disease(
        infection = Susceptible >> Exposed @ contacts*ss.by([0.05, 0.08], dims=['age'])*Infected,
        onset     = Exposed >> Infected @ ss.by([1/3, 1/4], dims=['age']),
        recovery  = Infected >> Recovered @ 1/6,
        init_n    = dict(Exposed=ss.by([[10, 0], [0, 0]], dims=['age', 'patch']), Infected=ss.by([[5, 0], [10, 0]], dims=['age', 'patch'])),
        name      = 'oseir',
    )
    totals = np.array([[6000, 2000], [4000, 1000]])
    seir.stratify(age=['0', '1'], fractions=[8/13, 5/13])
    seir.stratify(patch=['0', '1'], fractions=ss.by(totals/totals.sum(), dims=['age', 'patch']), mobility=ss.geo.Mobility(matrix=[[0, 0.05], [0.1, 0]], kind='commute'))
    ref = ss.Sim(seir, networks=ss.contacts([[4.0, 1.0], [1.0, 3.0]]), total_pop=13_000, unit='day', stop=ss.days(60), solver='ode', verbose=0).run()
    assert np.allclose(res.Infected.values, ref.results.oseir.Infected.values, rtol=1e-9), 'The import should match the model written by hand'
    return sim


def seir_sim(**kwargs):
    """ An SEIR to export """
    pars = dict(diseases=ss.SEIR(beta=ss.perday(0.4), dur_exp=ss.days(4), dur_inf=ss.days(8), init_prev=0.01), total_pop=10_000, unit='day', dur=100, solver='ode', verbose=0)
    return ss.Sim(**(pars | kwargs))


def test_memilio_export():
    sc.heading('Testing exporting to MEmilio')
    text, summary, lost = to_memilio(seir_sim())
    compile(text, 'seir.py', 'exec')
    assert 'memilio.simulation.oseir' in text and 'TimeExposed[A0] = 4.0' in text and 'TimeInfected[A0] = 8.0' in text
    assert 'TransmissionProbabilityOnContact[A0] = 0.4' in text

    # Contacts from a network; births and deaths are not in MEmilio's SIR; an SIS is none of its models
    text, summary, lost = to_memilio(seir_sim(diseases=ss.SEIR(beta=ss.perday(0.05), dur_exp=ss.days(4), dur_inf=ss.days(8)), networks=ss.RandomNet(n_contacts=8)))
    assert 'np.full((1, 1), 8.0)' in text and 'TransmissionProbabilityOnContact[A0] = 0.05' in text
    sir = seir_sim(diseases=ss.SIR(beta=ss.perday(0.3), dur_inf=ss.days(10)), demographics=[ss.birth(rate=ss.peryear(0.5)), ss.death(rate=ss.peryear(0.5))])
    text, summary, lost = to_memilio(sir)
    assert 'memilio.simulation.osir' in text
    assert sum('not exported: the flow' in line for line in lost) == 4, 'The births and the three deaths should be listed'
    with pytest.raises(NotImplementedError, match='osir'):
        to_memilio(seir_sim(diseases=ss.SIS(beta=ss.perday(0.3), dur_inf=ss.days(10))))
    return text


@pytest.mark.skipif(not has_memilio, reason='needs MEmilio\'s environment (STARSIM_LF)')
def test_memilio_roundtrip(tmp_path):
    sc.heading('Testing an export to MEmilio, run in MEmilio and read back')
    sim = seir_sim()
    text, summary, lost = to_memilio(sim)
    script, dumped = tmp_path/'seir.py', tmp_path/'seir.json'
    script.write_text(text)
    subprocess.run([memilio_python, ss.converters.memilio.__file__, str(script), str(dumped)], check=True, capture_output=True) # Runs the script, and records the model it simulates
    back = load_memilio(str(dumped), **kw).run()
    res = sim.run().results.seir
    pairs = dict(S='Susceptible', E='Exposed', I='Infected', R='Recovered')
    for a,b in pairs.items():
        assert np.allclose(back.results.oseir[b].values, res[a].values, rtol=1e-9), f'{a} should be the same once read back'

    # MEmilio's own run, as the script prints it
    out = subprocess.run([memilio_python, str(script)], check=True, capture_output=True, text=True).stdout.splitlines()
    rows = [line.split() for line in out if line[:1].isdigit()]
    head = next(line.split() for line in out if line.startswith('Day'))
    vals = np.array(rows, dtype=float)
    for a,b in pairs.items():
        assert np.allclose(vals[:, head.index(b)], res[a].values, rtol=0, atol=1e-4*res[a].values.max()), f'MEmilio\'s {b} should match the ODE'
    return back


if __name__ == '__main__':
    T = sc.timer()
    sim1 = test_seir()
    sim2 = test_secir()
    sim3 = test_graph_json()
    test_memilio_export()
    T.toc()
