"""
Test importing summer2 and Atomica models
"""
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import starsim.interop
import paths
from starsim.interop.summer2 import load_summer2, outputs
from starsim.interop.atomica import load_atomica

lfd = paths.lfd/'examples' # The original examples, if they are here
summer2_python = paths.lfd/'_envs/summer2/bin/python' # summer2's environment, to write a live model with dump_summer2.py
summer2_tb = '''
import sys, json
import numpy as np
import summer2
from summer2.parameters import Parameter
from summer2.functions.time import get_sigmoidal_interpolation_function

def build(): # The TB model of v4_design/oct17/switching/summer2.md: named Parameters, a sigmoidal scale-up of detection, and ages with a plain-array mixing matrix
    m = summer2.CompartmentalModel(times=[1950.0, 2025.0], compartments=list('SELITR'), infectious_compartments=['I'], timestep=1.0)
    m.set_initial_population(distribution={'S': 1e6 - 1000, 'I': 1000})
    beta = Parameter('contact_rate')
    m.add_infection_frequency_flow('infection', beta, 'S', 'E')
    m.add_infection_frequency_flow('reinfection_L', beta*Parameter('rr_reinf'), 'L', 'E')
    m.add_infection_frequency_flow('reinfection_R', beta*Parameter('rr_reinf'), 'R', 'E')
    for name,rate,src,dst in [('stabilisation', 2.0, 'E', 'L'), ('early_activation', 0.2, 'E', 'I'), ('late_activation', 0.002, 'L', 'I'), ('self_recovery', 0.2, 'I', 'R'), ('treatment_end', 2.0, 'T', 'R')]:
        m.add_transition_flow(name, rate, src, dst)
    m.add_transition_flow('detection', get_sigmoidal_interpolation_function([1950.0, 1990.0, 2020.0], [0.0, 0.5, 1.0])*Parameter('detect_max'), 'I', 'T')
    m.add_death_flow('tb_death', 0.2, 'I')
    m.add_universal_death_flows('universal_death', death_rate=1/70)
    m.add_crude_birth_flow('birth', 0.02, 'S')
    ages = ['0', '5', '15', '35', '50', '70']
    strat = summer2.AgeStratification('age', [int(a) for a in ages], list('SELITR'))
    strat.set_mixing_matrix(np.array([[3, 1, 1.5, 1, 0.5, 0.3], [1, 8, 2, 1.5, 0.8, 0.4], [1.5, 2, 5, 2.5, 1.5, 0.6], [1, 1.5, 2.5, 4, 2, 0.8], [0.5, 0.8, 1.5, 2, 3, 1.2], [0.3, 0.4, 0.6, 0.8, 1.2, 2]])/4)
    strat.set_flow_adjustments('early_activation', dict(zip(ages, [summer2.Multiply(3.0), summer2.Multiply(1.5), None, None, None, None])))
    strat.add_infectiousness_adjustments('I', dict(zip(ages, [summer2.Multiply(0.1), summer2.Multiply(0.2), None, None, None, None])))
    m.stratify_with(strat)
    m.request_output_for_flow('notifications', 'detection')
    return m

if __name__ == '__main__': # Write it with dump_summer2.py (from the folder given), then run it and write the compartments at the end and the notifications
    sys.path.insert(0, sys.argv[1])
    import dump_summer2
    pars = json.load(open(f'{sys.argv[2]}/pars.json'))
    dump_summer2.dump(build(), f'{sys.argv[2]}/tb.json', pars=pars)
    m = build()
    m.run(pars)
    comps = {str(c):float(v) for c,v in zip(m.compartments, m.outputs[-1])}
    json.dump(dict(comps=comps, notifications=m.get_derived_outputs_df().notifications.tolist()), open(f'{sys.argv[2]}/ref.json', 'w'))
'''
atomica_lib = paths.lfd/'_envs/atomica/lib/python3.12/site-packages/atomica/library' # Atomica's library workbooks, in its environment
atomica_demos = dict( # Atomica 1.32.4's own runs at dt=1 (start, stop, values at the end), from v4_design/oct17/switching/atomica.md
    udt      = (2016, 2023, dict(n_dx=2450.0, n_tx=3550.0)),
    dt       = (2016, 2024, dict(n_dx=240.0, n_tx=3360.0)),
    diabetes = (2014, 2035, dict(n_sus=1218222.500392, n_sc_uncomp=775.919855)),
    combined = (2000, 2023, {'SIR_0-4:n_rec':728.476002}), # The first population type only, as listed
    hiv_dyn  = (2016, 2035, dict(all_vs=1596938.31)), # A characteristic of one compartment, under its own name
)
kw = dict(verbose=False)


def summer2_model():
    """ summer2's SEIR by age with a clinical stratification of I only, as dump_summer2.py writes it """
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
    assert np.isclose(res.n_I.values[0], 10) and np.isclose(res.n_I.select(clinical='asymptomatic').values[0], 10), 'The infected should start asymptomatic, as in summer2'
    assert res.n_I.select(clinical='hospital').values[-1] > 0

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
        assert np.allclose(res[f'n_{state}'].values, hand[f'n_{state}'].values, atol=1e-6), f'n_{state} should match the hand translation'

    # Derived outputs, in summer2's conventions (a flow is a rate)
    out = outputs(sim)
    assert np.allclose(out.incidence_old, res.incidence.select(age='old').values/0.1)
    assert np.allclose(out.accum_old, np.cumsum(out.incidence_old))
    hosp = res.n_I.select(clinical='hospital').values
    assert np.allclose(out.prop_infected_hosp, hosp/res.n_I.values)

    # A mixing matrix, and a flow parameter that is a function, which is listed as not translated
    d = summer2_model()
    d['stratifications'] = d['stratifications'][:1]
    d['stratifications'][0]['mixing_matrix'] = [[0.2, 0.3], [0.5, 0.7]]
    d['flows'][2]['param'] = dict(function='(t * 0.01)')
    with sc.capture():
        sim2 = load_summer2(d, **kw)
    assert sim2.pars.networks is not None
    assert any('recovery' in name for name,why in sim2.imported.incomplete)

    # Against the original's results, if the example is here
    folder = f'{lfd}/summer2/seir-age-clinical'
    if os.path.exists(folder):
        ref = pd.read_csv(f'{folder}/reference/results.csv')
        theirs = ref[ref.series == 'I_old_hospital'].value.values
        assert np.abs(res.n_I.select(age='old', clinical='hospital').values - theirs).max() < 0.01, 'Should agree with summer2'

    # A live summer2 model with named Parameters, a sigmoidal scale-up, and ages with an array mixing matrix, written by dump_summer2.py in summer2's environment, against summer2's own run
    if summer2_python.exists():
        with tempfile.TemporaryDirectory() as tmp:
            tmp = sc.path(tmp)
            sc.savetext(tmp/'tb.py', summer2_tb)
            sc.savejson(tmp/'pars.json', dict(contact_rate=10.0, rr_reinf=0.21, detect_max=1.5))
            subprocess.run([str(summer2_python), '-I', str(tmp/'tb.py'), str(sc.path(ss.interop.__file__).parent), str(tmp)], check=True, capture_output=True)
            tb = load_summer2(str(tmp/'tb.json'), **kw)
            ref = sc.loadjson(tmp/'ref.json')
        assert not tb.imported.incomplete
        tbres = tb.run().results.summer2
        levels = dict(zip(['0', '5', '15', '35', '50', '70'], ['0-4', '5-14', '15-34', '35-49', '50-69', '70+'])) # summer2's age groups, as Starsim labels them
        for c,val in ref['comps'].items():
            state, age = c.split('Xage_')
            ours = tbres[f'n_{state}'].select(age=levels[age]).values[-1]
            assert np.isclose(ours, val, rtol=1e-3), f'{c} in 2025 should agree with summer2' # The sigmoid is sampled every tenth of a step
        notif = outputs(tb).notifications
        for year in [1990, 2010, 2024]:
            assert np.isclose(notif[year - 1950], ref['notifications'][year - 1950], rtol=0.005), f'Notifications in {year} should agree with summer2' # summer2's is the mean of the rates at either end of the year
    return sim


def atomica_files(folder):
    """ A small Atomica framework and databook: an SIR with a probability, a duration, a rate, a number, and a function """
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
    assert np.allclose([res.n_sus[0], res.n_inf[0], res.n_rec[0]], [900, 50, 50]), 'The initial numbers should be solved from the characteristics'
    assert sim.t.yearvec[-1] == 2009, 'Atomica runs to five years after the data, if there are no births'

    # Atomica's step by hand: rates and probabilities move value × dt, durations dt/value, numbers value × dt people
    x = np.array([900.0, 50, 50, 0])
    vacc = np.interp(np.arange(2000, 2010), [2000, 2001, 2002, 2003, 2004], [10, 10, 20, 20, 20])
    for i,t in enumerate(range(2000, 2009)):
        s, inf, r, d = x
        prev = inf/(s + inf + r)
        flows = dict(si=0.6*prev*s, sr=min(vacc[i], s), ir=inf/5, id=0.02*inf)
        x = np.array([s - flows['si'] - flows['sr'], inf + flows['si'] - flows['ir'] - flows['id'], r + flows['sr'] + flows['ir'], d + flows['id']])
    assert np.allclose([res.n_sus[-1], res.n_inf[-1], res.n_rec[-1], res.n_dead[-1]], x), 'Should match Atomica\'s Euler step'
    assert np.isclose(res.ch_prev[0], 0.05)

    # Against the original's results, if the example is here
    folder = f'{lfd}/atomica/sir'
    if os.path.exists(folder):
        sim2 = load_atomica(f'{folder}/original/sir_framework.xlsx', f'{folder}/original/sir_databook.xlsx', dt=1, **kw).run()
        ref = pd.read_csv(f'{folder}/reference/results.csv')
        for comp in ['sus', 'inf', 'rec', 'dead']:
            theirs = ref[ref.series == f'adults_{comp}'].value.values
            assert np.allclose(sim2.results.SIR[f'n_{comp}'].values, theirs), f'{comp} should agree with Atomica'

    # Atomica's own library workbooks, if its environment is here: lower-case headings, numbers in the databook, the setup weights, a timed duration, several population types
    if atomica_lib.exists():
        for demo,(start,stop,ref) in atomica_demos.items():
            sim3 = load_atomica(atomica_lib/f'{demo}_framework.xlsx', atomica_lib/f'{demo}_databook.xlsx', start=start, stop=stop, dt=1, **kw)
            sim3.pars.verbose = 0
            res = sim3.run().results[sim3.diseases[0].name]
            for key,val in ref.items():
                ours = res[key].values[-1] if ':' not in key else res[key.split(':')[1]].select(population=key.split(':')[0]).values[-1]
                assert np.isclose(ours, val, rtol=1e-6), f'{demo}: {key} is {ours} in {stop}, against {val} in Atomica'
    return sim


if __name__ == '__main__':
    T = sc.timer()
    sim1 = test_summer2()
    sim2 = test_atomica()
    T.toc()
