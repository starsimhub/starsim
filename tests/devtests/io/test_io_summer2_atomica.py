"""
Test importing summer2 and Atomica models, and exporting to them (run in their own environments, `_envs/` in the lingua_franca repo, if they are there)
"""
import os
import json
import tempfile
import subprocess
import warnings
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import paths
from starsim.converters.summer2 import load_summer2, outputs, to_summer2
from starsim.converters.atomica import load_atomica, to_atomica

kw = dict(verbose=False)
summer2_python = paths.lf/'_envs'/'summer2'/'bin'/'python'
atomica_python = paths.lf/'_envs'/'atomica'/'bin'/'python'
has_atomica = atomica_python.exists()


def sir_sim():
    """ An SIR, in days """
    S, I, R = ss.states('S I R')
    dis = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3) * I), recovery=ss.Transition(I, R, dur=ss.days(7)), init_prev=0.01, name='sir')
    return ss.Sim(dis, total_pop=1000, unit='day', stop=60, verbose=0)


def seird_sim():
    """ Births, deaths from every state, a branch, and a lognormal dwell time, which is written as Erlang stages """
    S, E, I, R, D = ss.states('S E I R D')
    dis = ss.Disease(
        births    = ss.birth(S, rate=ss.peryear(0.02)),
        infection = ss.Transition(S, E, rate=ss.perday(0.4) * I),
        onset     = ss.Transition(E, I, dur=ss.days(4)),
        outcome   = ss.Transition(I, {R: None, D: 0.1}, dur=ss.lognormal(ss.days(8), ss.days(3))),
        deaths    = ss.death([S, E, I, R], rate=ss.peryear(0.01)),
        init_prev = 0.01, name='seird')
    return ss.Sim(dis, total_pop=1e5, unit='day', stop=120, verbose=0)


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
        incidence       = ss.Transition(E, I, rate=0.5*ss.by(age=[1, 1.1])),
        recovery        = ss.Transition(I, R, rate=0.3),
        infection_death = ss.death(I, rate=0.05*ss.by(clinical=[0, 1, 1, 2])),
        rel_trans_by_clinical = ss.modify('seir.rel_trans', multiply=ss.by(clinical=[0.5, 1, 0.2, 0.1])),
        init_n          = dict(I=ss.by([[5, 0, 0, 0], [5, 0, 0, 0]], dims=['age', 'clinical'])),
        name            = 'seir',
    )
    seir.stratify(age=['young', 'old'], clinical=ss.levels(['asymptomatic', 'symptomatic', 'isolated', 'hospital'], fractions=ss.by([[0.4, 0.3, 0.2, 0.1], [0.3, 0.3, 0.2, 0.3]], dims=['age', 'clinical']), states=['I']))
    hand = ss.Sim(seir, total_pop=1000, dt=0.1, stop=20, verbose=0).run().results.seir
    for state in 'SEIR':
        assert np.allclose(res[state].values, hand[state].values, atol=1e-6), f'{state} should match the hand translation'

    # Derived outputs, in summer2's conventions (a flow is a rate)
    out = outputs(sim)
    assert np.allclose(out.incidence_old, np.concatenate([[0], res.incidence.select(age='old').values[:-1]])/0.1) # At the end of each step, as summer2 reports it
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


def test_summer2_export():
    sc.heading('Testing export to summer2')
    text, summary, lost = to_summer2(seird_sim()) # Births, deaths, a branch, and Erlang stages
    compile(text, 'seird.py', 'exec')
    for call in ['add_crude_birth_flow', 'add_death_flow', 'add_infection_frequency_flow', 'add_transition_flow']:
        assert f'model.{call}(' in text
    assert "infectious_compartments=['I_1', 'I_2', 'I_3', 'I_4', 'I_5', 'I_6', 'I_7']" in text, 'Each stage of I should be infectious'
    assert any('7 exponential stages' in line for line in lost)

    # A rate of no summer2 kind, and one that varies in time, are listed
    S, I, R = ss.states('S I R')
    dis = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.5)*I.frac*np.maximum(np.exp(-2*I.frac), 0.2)), recovery=ss.Transition(I, R, rate=ss.seasonal(ss.perday(0.1), amplitude=0.2)), init_prev=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        text, summary, lost = to_summer2(ss.Sim(dis, total_pop=1000, unit='week', stop=10, verbose=0))
    assert sum('not exported: the flow' in line for line in lost) == 2 and any('varies in time' in line for line in lost)
    assert any('which load_summer2() reads as days' in line for line in lost)
    return text


@pytest.mark.skipif(not summer2_python.exists(), reason='needs summer2\'s environment (_envs/summer2 in the lingua_franca repo; set STARSIM_LF)')
def test_summer2_export_run(tmp_path):
    sc.heading('Testing export to summer2, run in summer2 and read back')
    dumper = sc.path(ss.__file__).parent/'converters'/'summer2.py'
    run = 'import sys, json, runpy; m = runpy.run_path(sys.argv[1])["model"]; m.run(); df = m.get_outputs_df(); json.dump({c:df[c].tolist() for c in df.columns}, open(sys.argv[2], "w"))'
    for make in [sir_sim, seird_sim]:
        sim = make()
        script, dumped, ran = [tmp_path/f'{make.__name__}.{ext}' for ext in ['py', 'json', 'out.json']]
        sc.savetext(script, to_summer2(sim)[0])
        for args in [[dumper, script, dumped], ['-c', run, script, ran]]: # The JSON that load_summer2() reads, and summer2's own run
            out = subprocess.run([str(summer2_python), *map(str, args)], capture_output=True, text=True, cwd=tmp_path, timeout=300)
            assert out.returncode == 0, out.stderr
        back = load_summer2(str(dumped), **kw)
        assert back.imported.incomplete == []
        sim.run(); back.run()
        ours, again, theirs = sim.results[sim.diseases[0].name], back.results.summer2, json.load(open(ran))
        for state in ['S', 'R']:
            assert np.allclose(ours[state], again[state], rtol=1e-9, atol=1e-9), f'{state} should be the same read back'
            assert np.allclose(ours[state], theirs[state], rtol=1e-4, atol=1e-3), f'{state} should be the same in summer2'
    return back


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
    assert sim._kernel_reason() is None, 'The whole run should be taken in one call to the kernels'
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


def sir_births():
    """ An SIR with births and deaths, and a gamma dwell time (as Erlang stages), to export; stepped as Atomica steps, by Euler's method """
    sir = ss.SIR(beta=ss.perday(0.3), dur_inf=ss.gamma(shape=2, scale=ss.days(5)), init_prev=0.01)
    return ss.Sim(diseases=sir, demographics=[ss.birth(rate=ss.peryear(0.5)), ss.death(rate=ss.peryear(0.5))], total_pop=10_000, start=2000, stop=2000.25, dt=ss.days(1),
                  solver=ss.solvers.ode(integrator='euler', substeps=1), verbose=0)


def infected(res):
    """ The infected, summed over their stages """
    return sum(res[k].values for k in res.keys() if k.startswith('I_') and k[2:].isdigit())


def test_atomica_export(tmp_path):
    sc.heading('Testing exporting to Atomica, and reading it back')
    sim = sir_births()
    data, summary, lost = to_atomica(sim)
    fw = tmp_path/'sir_framework.xlsx'
    fw.write_bytes(data)
    trans = pd.read_excel(fw, sheet_name='Transitions', header=None)
    assert {'births', 'deaths'} <= set(trans.iloc[0]), 'Births come from a source, and deaths go to a sink'
    assert any('Euler' in line for line in lost)
    back = load_atomica(str(fw), start=2000, stop=2000.25, dt=1/365, **kw).run() # The framework alone: its defaults
    res = sim.run().results.sir
    assert np.allclose(infected(back.results.sir), res.I.values, rtol=1e-9), 'The infected should be the same once read back'
    assert np.allclose(back.results.sir.S.values, res.S.values, rtol=1e-9)
    return back


@pytest.mark.skipif(not has_atomica, reason='needs Atomica\'s environment (STARSIM_LF)')
def test_atomica_framework(tmp_path):
    sc.heading('Testing an export to Atomica, run in Atomica')
    sim = sir_births()
    fw = tmp_path/'sir_framework.xlsx'
    fw.write_bytes(to_atomica(sim)[0])
    code = 'import json, logging, numpy as np, atomica as at; at.logger.setLevel(logging.WARNING); ' \
           f'F = at.ProjectFramework({str(fw)!r}); D = at.ProjectData.new(F, np.arange(2000, 2001), pops={{"pop": "Population"}}, transfers=0); ' \
           'P = at.Project(framework=F, databook=D.to_spreadsheet(), do_run=False); P.settings.update_time_vector(start=2000, end=2000.25, dt=1/365); ' \
           'pop = P.run_sim().model.pops[0]; print(json.dumps({c.name: c.vals.tolist() for c in pop.comps}))'
    out = subprocess.run([atomica_python, '-c', code], check=True, capture_output=True, text=True)
    comps = json.loads(out.stdout.splitlines()[-1])
    res = sim.run().results.sir
    n = len(res.S.values)
    assert np.allclose(np.array(comps['S'][:n]), res.S.values, rtol=1e-9), 'Atomica should step the model as Starsim\'s Euler method does'
    assert np.allclose(sum(np.array(v[:n]) for k,v in comps.items() if k.startswith('I_')), res.I.values, rtol=1e-9)
    return comps


if __name__ == '__main__':
    T = sc.timer()
    sim1 = test_summer2()
    text = test_summer2_export()
    sim2 = test_atomica()
    sim3 = test_atomica_junction()
    sim4 = test_atomica_export(sc.path(tempfile.mkdtemp()))
    T.toc()
