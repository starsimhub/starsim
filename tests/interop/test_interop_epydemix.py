"""
Test importing epydemix models
"""
import os
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss
import paths
from starsim.interop.epydemix import load_epydemix

lfd = paths.lfd/'examples' # The original examples, if they are here
kw = dict(verbose=False)


def seird_model():
    """ An SEIR with deaths in two age groups, as dump_epydemix.py writes it """
    days = 61
    return dict(
        format = 'epydemix',
        name = 'seird',
        compartments = ['S', 'E', 'I', 'R', 'D'],
        transitions = [
            dict(source='S', target='E', kind='mediated', params=['beta*r', 'I']),
            dict(source='E', target='I', kind='spontaneous', params='eps'),
            dict(source='I', target='R', kind='spontaneous', params='mu*(1-IFR)'),
            dict(source='I', target='D', kind='spontaneous', params='mu*IFR'),
        ],
        parameters = dict(
            beta = dict(kind='number', value=0.3),
            r    = dict(kind='time', value=[1.0]*20 + [0.5]*(days - 20)), # Contacts halve on day 20
            eps  = dict(kind='number', value=0.25),
            mu   = dict(kind='number', value=0.2),
            IFR  = dict(kind='age', value=[0.01, 0.1]),
        ),
        overrides = dict(eps=[dict(start_date='2020-02-01', end_date='2020-02-10', value=dict(kind='number', value=0.5))]),
        interventions = [dict(layer='school', start_date='2020-01-15', end_date='2020-02-15', name='closure', reduction_factor=0.0, new_matrix=None)],
        population = dict(name='Toy', groups=['0-19', '20+'], Nk=[6000.0, 4000.0], contacts=dict(home=[[2.0, 1.0], [1.5, 2.0]], school=[[3.0, 0.0], [0.0, 0.0]])),
        run = dict(start_date='2020-01-01', end_date='2020-03-01', dt=1.0, initial=dict(S=[5990.0, 3990.0], E=[0.0, 0.0], I=[10.0, 10.0], R=[0.0, 0.0], D=[0.0, 0.0]), parameters={}),
    )


def test_epydemix():
    sc.heading('Testing importing an epydemix model')
    with pytest.warns(ss.errors.ImportIncomplete): # The school closure is listed, not translated
        sim = load_epydemix(seird_model(), solver='ode', **kw)
    assert [k for k,_ in sim.imported.incomplete] == ['intervention closure']
    sim.run()
    dis = sim.diseases[0]
    assert list(dis.transitions.keys()) == ['S_to_E', 'E_to_I', 'I_to_D_or_R'], 'The two exits from I should be one branch'
    res = sim.results.epydemix
    assert res.n_I.select(age=0).values[0] == 10, 'Each group should start with its initial numbers'
    total = sum(res[f'n_{c}'].values for c in 'SEIRD')
    assert np.allclose(total, 10_000)

    # The same model by hand
    S, E, I, R, D = ss.states(5)
    r = ss.timeseries({'2020-01-01': 1.0, '2020-01-21': 0.5})
    eps = ss.timeseries({'2020-01-01': 0.25, '2020-02-01': 0.5, '2020-02-11': 0.25})
    seird = ss.Disease(
        S_to_E = S >> E @ 0.3*r*I,
        E_to_I = E >> I @ eps,
        I_to_D_or_R = ss.Transition(I, [D, R], rate=0.2, p=[ss.by([0.01, 0.1], dims=['age']), None]),
        init_n = dict(I=ss.by([10, 10], dims=['age'])),
        name = 'epydemix',
    )
    seird.stratify(age=[0, 20], split=[0.6, 0.4])
    ref = ss.Sim(seird, networks=ss.contacts(age=[[5.0, 1.0], [1.5, 2.0]]), total_pop=10_000, unit='day', start='2020-01-01', stop='2020-03-02',
                 solver='ode', substeps=1, use_aging=False, verbose=0).run()
    for c in 'SEIRD':
        ours, theirs = res[f'n_{c}'].values, ref.results.epydemix[f'n_{c}'].values
        assert np.allclose(ours, theirs, rtol=1e-9), f'{c} should match the model written by hand'
    return sim


def test_kenya():
    sc.heading('Testing importing the epydemix SIR in Kenya against the original')
    folder = f'{lfd}/epydemix/sir_kenya/reference'
    if not os.path.exists(folder):
        pytest.skip(paths.no_lfd)
    pop = sc.loadjson(f'{folder}/population.json')
    d = dict(
        format = 'epydemix',
        name = 'sir',
        compartments = ['Susceptible', 'Infected', 'Recovered'],
        transitions = [dict(source='Susceptible', target='Infected', kind='mediated', params=['transmission_rate', 'Infected']),
                       dict(source='Infected', target='Recovered', kind='spontaneous', params='recovery_rate')],
        parameters = dict(transmission_rate=dict(kind='number', value=0.04), recovery_rate=dict(kind='number', value=0.1)),
        population = dict(name='Kenya', groups=pop['groups'], Nk=pop['Nk'], contacts=pop['contacts']),
        run = dict(start_date=pop['start'], end_date=pop['end'], dt=1.0, initial=pop['initial']),
    )
    sim = load_epydemix(d, **kw)
    msim = ss.parallel(sim, n_runs=200) # Restarted in this process; enough runs that the mean is within about 1% of the true one
    ours = msim.results.epydemix.n_Infected.mean() # Over the runs
    ref = pd.read_csv(f'{folder}/results.csv')
    theirs = ref[ref.series == 'Infected_total'].sort_values('time').value.values
    diff = np.abs(ours - theirs).max()/theirs.max()
    assert diff < 0.03, f'Expected the mean number infected within 3% of the original, not {diff:.1%}'
    return sim


if __name__ == '__main__':
    T = sc.timer()
    sim1 = test_epydemix()
    sim2 = test_kenya()
    T.toc()
