"""
Test importing summer2 and Atomica models
"""
import os
import tempfile
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from starsim.converters.summer2 import load_summer2, outputs
from starsim.converters.atomica import load_atomica

kw = dict(verbose=False)


def summer2_model():
    """ summer2's SEIR by age with a clinical stratification of I only, as summer2.py writes it """
    ages = ['young', 'old']
    clinical = ['asymptomatic', 'symptomatic', 'isolated', 'hospital']
    mult = lambda *v: dict(zip(clinical, [None if x is None else ['multiply', x] for x in v]))
    initial = [dict(name=c, strata=dict(age=a), value=495.0 if c == 'S' else 0.0) for c in 'SER' for a in ages]
    initial += [dict(name='I', strata=dict(age=a, clinical=c), value=5.0 if c == 'asymptomatic' else 0.0) for a in ages for c in clinical]
    return dict(
        format = 'summer2',
        compartments = ['S', 'E', 'I', 'R'],
        infectious = ['I'],
        initial_population = dict(S=990, I=10, E=0, R=0),
        initial = initial,
        times = [0, 20],
        timestep = 0.1,
        flows = [
            dict(name='infection', kind='InfectionFrequencyFlow', sources=['S'], dest='E', param=2.0),
            dict(name='incidence', kind='TransitionFlow', sources=['E'], dest='I', param=0.5),
            dict(name='recovery', kind='TransitionFlow', sources=['I'], dest='R', param=0.3),
            dict(name='infection_death', kind='DeathFlow', sources=['I'], dest=None, param=0.05),
        ],
        stratifications = [
            dict(name='age', strata=ages, compartments=['S', 'E', 'I', 'R'], population_split=dict(young=0.5, old=0.5), flow_adjustments={},
                 infectiousness_adjustments={}, mixing_matrix=None, ageing=False, strain=False),
            dict(name='clinical', strata=clinical, compartments=['I'], population_split=dict(asymptomatic=1.0, symptomatic=0.0, isolated=0.0, hospital=0.0),
                 flow_adjustments=dict(
                     incidence = [dict(adjustments=mult(0.4, 0.3, 0.2, 0.1), source_strata=dict(age='young'), dest_strata={}),
                                  dict(adjustments=mult(0.3, 0.3, 0.2, 0.3), source_strata=dict(age='old'), dest_strata={})],
                     infection_death = [dict(adjustments={**mult(None, None, None, 2.0), 'asymptomatic':['overwrite', 0.0]}, source_strata={}, dest_strata={})],
                 ),
                 infectiousness_adjustments=dict(I=mult(0.5, None, 0.2, 0.1)), mixing_matrix=None, ageing=False, strain=False),
        ],
        outputs = dict(
            incidence_old = dict(request_type='flow', flow_name='incidence', source_strata=dict(age='old'), dest_strata={}),
            hospital_occupancy = dict(request_type='comp', compartments=['I'], strata=dict(clinical='hospital')),
            all_infected = dict(request_type='comp', compartments=['I'], strata={}),
            prop_infected_hosp = dict(request_type='param_func', func='(hospital_occupancy / all_infected)'),
            accum_old = dict(request_type='cum', source='incidence_old'),
        ),
    )


def test_summer2():
    sc.heading('Testing importing a summer2 model')

    # The import: each flow kind, the stratifications, and the derived outputs
    sim = load_summer2(summer2_model(), **kw)
    assert sim.imported.incomplete == []
    sim.run()
    res = sim.results.summer2
    assert np.isclose(res.I.values[0], 10) and np.isclose(res.I.select(clinical='asymptomatic').values[0], 10), 'The infected should start asymptomatic, as in summer2'
    assert res.I.select(clinical='hospital').values[-1] > 0

    # The same model written by hand, as in the vignette
    S, E, I, R = ss.states('S E I R')
    seir = ss.Disease(
        infection       = ss.Transition(S, E, rate=2 * I),
        incidence       = ss.Transition(E, I, rate=0.5),
        recovery        = ss.Transition(I, R, rate=0.3),
        infection_death = ss.death(I, rate=0.05),
        init_n          = dict(I=ss.by([[5, 0, 0, 0], [5, 0, 0, 0]], dims=['age', 'clinical'])),
    )
    seir.stratify(age=['young', 'old'], adjust={'incidence': [1, 1.1]})
    seir.stratify(clinical=['asymptomatic', 'symptomatic', 'isolated', 'hospital'], states=['I'],
                  split=ss.by([[0.4, 0.3, 0.2, 0.1], [0.3, 0.3, 0.2, 0.3]], dims=['age', 'clinical']),
                  adjust={'infection_death': [0, 1, 1, 2], 'rel_trans': [0.5, 1, 0.2, 0.1]})
    hand = ss.Sim(seir, total_pop=1000, dt=0.1, stop=20, verbose=0).run().results.seir
    for state in 'SEIR':
        assert np.allclose(res[state].values, hand[state].values, atol=1e-6), f'{state} should match the hand translation'

    # Derived outputs, in summer2's conventions (a flow is a rate)
    out = outputs(sim)
    assert np.allclose(out.incidence_old, res.incidence.select(age='old').values/0.1)
    assert np.allclose(out.accum_old, np.cumsum(out.incidence_old))
    hosp = res.I.select(clinical='hospital').values
    assert np.allclose(out.prop_infected_hosp, hosp/res.I.values)

    # A mixing matrix, and a flow parameter that is a function, which is listed as not translated
    d = summer2_model()
    d['stratifications'] = d['stratifications'][:1]
    d['stratifications'][0]['mixing_matrix'] = [[0.2, 0.3], [0.5, 0.7]]
    d['flows'][2]['param'] = dict(function='(t * 0.01)')
    with sc.capture():
        sim2 = load_summer2(d, **kw)
    assert sim2.pars.networks is not None
    assert any('recovery' in name for name,why in sim2.imported.incomplete)
    return sim


def atomica_files(folder, junction=False):
    """ A small Atomica framework and databook: an SIR with a probability, a duration, a rate, a number, and a function; with junction=True, those infected pass through a junction, to inf for a share that rises with prevalence and to rec for the rest """
    fw = os.path.join(folder, 'framework.xlsx')
    db = os.path.join(folder, 'databook.xlsx')
    sheets = dict(
        About = [['Name', 'Description'], ['tiny', 'A tiny SIR']],
        Compartments = [['Code Name', 'Display Name', 'Is Source', 'Is Sink', 'Is Junction', 'Databook Page'],
                        ['sus', 'Susceptible', 'n', 'n', 'n', 'state_variables'], ['inf', 'Infected', 'n', 'n', 'n', None],
                        ['rec', 'Recovered', 'n', 'n', 'n', None], ['dead', 'Dead', 'n', 'y', 'n', None]],
        Characteristics = [['Code Name', 'Display Name', 'Databook Page', 'Components', 'Denominator'],
                           ['ch_all', 'Total', 'state_variables', 'sus, inf, rec', None], ['ch_prev', 'Prevalence', 'state_variables', 'inf', 'ch_all']],
        Parameters = [['Code Name', 'Display Name', 'Format', 'Databook Page', 'Default Value', 'Function'],
                      ['beta', 'Transmission probability', 'Probability', 'parameters', None, None],
                      ['foi', 'Force of infection', 'Rate', None, None, 'beta * ch_prev'],
                      ['dur', 'Duration of infection', 'Duration', 'parameters', None, None],
                      ['mort', 'Death rate', 'Rate', 'parameters', 0.02, None],
                      ['vacc', 'Number vaccinated', 'Number', 'parameters', None, None]],
        Transitions = [[None, 'sus', 'inf', 'rec', 'dead'], ['sus', None, 'foi', 'vacc', None], ['inf', None, None, 'dur', 'mort'], ['rec', None, None, None, None], ['dead', None, None, None, None]],
    )
    if junction:
        sheets['Compartments'].append(['jn', 'Junction', 'n', 'n', 'y', None])
        sheets['Parameters'].append(['p_sym', 'Share symptomatic', 'Proportion', None, None, 'min(10 * ch_prev, 1)'])
        sheets['Transitions'] = [[None, 'sus', 'inf', 'rec', 'dead', 'jn'], ['sus', None, None, 'vacc', None, 'foi'], ['inf', None, None, 'dur', 'mort', None],
                                 ['rec', None, None, None, None, None], ['dead', None, None, None, None, None], ['jn', None, 'p_sym', '>', None, None]]
    with pd.ExcelWriter(fw) as w:
        for name,rows in sheets.items():
            pd.DataFrame(rows).to_excel(w, sheet_name=name, header=False, index=False)
    years = list(range(2000, 2005))
    block = lambda label, units, const, vals: [[label, 'Provenance', 'Units', 'Uncertainty', 'Constant', None, *years],
                                              ['adults', None, units, None, const, 'OR', *vals], [None]*(6 + len(years))]
    state = block('Susceptible', 'Number', None, [900]*5) + block('Total', 'Number', None, [1000]*5) + block('Prevalence', 'Fraction', None, [0.05]*5)
    par = block('Transmission probability', 'Probability', 0.6, [None]*5) + block('Duration of infection', 'Duration', 5, [None]*5) + block('Number vaccinated', 'Number', None, [10, 10, 20, 20, 20])
    with pd.ExcelWriter(db) as w:
        pd.DataFrame([['Abbreviation', 'Full Name'], ['adults', 'Adults']]).to_excel(w, sheet_name='Population Definitions', header=False, index=False)
        pd.DataFrame(state).to_excel(w, sheet_name='State Variables', header=False, index=False)
        pd.DataFrame(par).to_excel(w, sheet_name='Parameters', header=False, index=False)
    return fw, db


def test_atomica():
    sc.heading('Testing importing an Atomica model')
    with tempfile.TemporaryDirectory() as tmp:
        fw, db = atomica_files(tmp)
        sim = load_atomica(fw, db, dt=1, **kw) # Initial numbers solved from the databook, and each format of parameter as Atomica steps it
    sim.run()
    res = sim.results.tiny
    assert np.allclose([res.sus[0], res.inf[0], res.rec[0]], [900, 50, 50]), 'The initial numbers should be solved from the characteristics'
    assert sim.t.yearvec[-1] == 2009, 'Atomica runs to five years after the data, if there are no births'

    # Atomica's step by hand: rates and probabilities move value × dt, durations dt/value, numbers value × dt people
    x = np.array([900.0, 50, 50, 0])
    vacc = np.interp(np.arange(2000, 2010), [2000, 2001, 2002, 2003, 2004], [10, 10, 20, 20, 20])
    for i,t in enumerate(range(2000, 2009)):
        s, inf, r, d = x
        prev = inf/(s + inf + r)
        flows = dict(si=0.6*prev*s, sr=min(vacc[i], s), ir=inf/5, id=0.02*inf)
        x = np.array([s - flows['si'] - flows['sr'], inf + flows['si'] - flows['ir'] - flows['id'], r + flows['sr'] + flows['ir'], d + flows['id']])
    assert np.allclose([res.sus[-1], res.inf[-1], res.rec[-1], res.dead[-1]], x), 'Should match Atomica\'s Euler step'
    assert np.isclose(res.ch_prev[0], 0.05)
    return sim


def test_atomica_junction():
    sc.heading('Testing an Atomica model whose junction splits by a share that changes with the counts')
    with tempfile.TemporaryDirectory() as tmp:
        fw, db = atomica_files(tmp, junction=True)
        sim = ss.converters.convert(fw, format='atomica', databook=db, dt=1, **kw) # Read, and the code saved beside the framework
        code = sim.imported.code
        assert sim.imported.filename.read_text() == code
    assert '\nch_prev = ' in code and code.count('ch_prev') > code.count('\nch_prev = ') + 1, 'The characteristic should be written once and used by name'
    assert 'sus_to_jn_inf' in code and 'sus_to_jn_rec' in code, 'A junction whose shares change should be a transition to each destination, with its share in the rate'
    sim.init()
    assert sim.kernel_reason() is None, 'The whole run should be taken in one call to the kernels'
    sim.run()
    res = sim.results.tiny

    # Atomica's step by hand, with those infected split at the junction
    x = np.array([900.0, 50, 50, 0])
    vacc = np.interp(np.arange(2000, 2010), [2000, 2001, 2002, 2003, 2004], [10, 10, 20, 20, 20])
    for i,t in enumerate(range(2000, 2009)):
        s, inf, r, d = x
        prev = inf/(s + inf + r)
        new = 0.6*prev*s
        sym = min(10*prev, 1)
        flows = dict(si=new*sym, sj=new*(1 - sym), sr=min(vacc[i], s), ir=inf/5, id=0.02*inf)
        x = np.array([s - new - flows['sr'], inf + flows['si'] - flows['ir'] - flows['id'], r + flows['sj'] + flows['sr'] + flows['ir'], d + flows['id']])
    assert np.allclose([res.sus[-1], res.inf[-1], res.rec[-1], res.dead[-1]], x), 'Should match Atomica\'s Euler step through the junction'
    return sim


if __name__ == '__main__':
    T = sc.timer()
    sim1 = test_summer2()
    sim2 = test_atomica()
    sim3 = test_atomica_junction()
    T.toc()
