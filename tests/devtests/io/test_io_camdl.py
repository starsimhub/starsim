"""
Test importing camdl models from camdl's IR (the tests against camdl itself are downstream, in the lingua_franca repo), and exporting sims as camdl models

The exports are compiled and run with camdl from its environment in the lingua_franca repo (`_envs/camdl`; set STARSIM_LF to say where the repo is), and skipped if it is not there.
"""
import subprocess
import numpy as np
import sciris as sc
import pytest
import starsim as ss
import paths
from starsim.converters.camdl import load_camdl, to_camdl

kw = dict(verbose=False)
camdl_bin = paths.lf/'_envs'/'camdl'/'bin'
has_camdl = (camdl_bin/'camdlc').exists() and (camdl_bin/'camdl').exists()

def op(name, left, right):
    return {'bin_op': {'op': name, 'left': left, 'right': right}}

par, pop = lambda name: {'param': name}, lambda name: {'pop': name}

# The IR camdlc writes for docs/models/camdl/sir/original/sir.camdl in the lingua_franca repo, trimmed of what the importer does not read
sir_ir = {'ir_version': '0.40', 'model': {
    'name': 'sir',
    'time_unit': 'days',
    'compartments': [{'name': 'S'}, {'name': 'I'}, {'name': 'R'}],
    'transitions': [
        {'name': 'infection', 'stoichiometry': [['S', -1], ['I', 1]], 'rate': op('div', op('mul', op('mul', par('beta'), pop('S')), pop('I')), {'binding_ref': 'N'})},
        {'name': 'recovery', 'stoichiometry': [['I', -1], ['R', 1]], 'rate': op('mul', par('gamma'), pop('I'))},
    ],
    'observations': [{'name': 'cases', 'emit_schedule': {'regular': {'start': 0.0, 'step': 7.0, 'end': 84.0}}, 'covers': {'kind': 'until', 'offset': 0.0, 'span': 7.0},
                      'projection': {'cumulative_flow': 'infection'},
                      'likelihood': {'neg_binomial': {'mean': {'expr': op('mul', par('rho'), {'projected': None})}, 'dispersion': {'expr': par('k')}}}}],
    'parameters': [
        {'name': 'beta', 'value': {'mode': 'estimated', 'bounds': [0.05, 1.0], 'prior': {'dist': {'log_normal': {'mu': -1.0, 'sigma': 0.5}}}}},
        {'name': 'gamma', 'value': {'mode': 'estimated', 'bounds': [0.01, 0.5], 'prior': {'dist': {'log_normal': {'mu': -2.0, 'sigma': 0.5}}}}},
        {'name': 'rho', 'value': {'mode': 'required'}},
        {'name': 'k', 'value': {'mode': 'required'}},
    ],
    'initial_conditions': {'S': {'deterministic': {'const': 9990.0}}, 'I': {'deterministic': {'const': 10.0}}},
    'simulation': {'t_start': 0.0, 't_end': 84.0, 'dt': None},
    'bindings': [{'name': 'N', 'expr': {'pop_sum': ['S', 'I', 'R']}}],
}, 'docs': {'parameters': {'beta': {'text': 'transmission rate'}}}}
sir_pars = dict(beta=0.4, gamma=0.15, rho=0.6, k=10)


def test_camdl_ir():
    """ The SIR from its IR: exposure, the outcome with its likelihood, and the priors, the same as the model written by hand """
    sc.heading('Testing camdl import from the IR')
    sim = load_camdl(sir_ir, pars=sir_pars, **kw)
    sim.run()
    tr = sim.diseases.sir.graph.transitions
    assert str(tr.infection.rate) == 'beta * I', 'infection should be read as exposure'
    pars = sim.diseases.sir.graph.pars
    assert pars.beta.bounds == [0.05, 1.0] and isinstance(pars.beta.prior, ss.lognormal) and pars.beta.prior.log_form, 'bounds and priors should be kept'
    assert pars.beta.doc == 'transmission rate'
    assert isinstance(sim.diseases.sir.graph.outcomes.cases.likelihood, ss.nbinom), 'the likelihood should be kept'

    # By hand, as in the camdl SIR translation in the lingua_franca repo
    beta, gamma, rho, k = ss.par(0.4), ss.par(0.15), ss.par(0.6), ss.par(10)
    S, I, R = ss.states(3)
    infection = S >> I @ beta * I
    cases = ss.outcome(rho * infection)
    sir = ss.Disease(infection, I >> R @ gamma, cases, init_n=dict(I=10), name='sir')
    ref = ss.Sim(sir, total_pop=10_000, unit='day', stop=ss.weeks(12), solver='ode', substeps=1, verbose=0).run() # One RK4 step a day, as camdl takes
    for key in ['I', 'R', 'infection', 'cases']:
        assert np.allclose(sim.results.sir[key].values, ref.results.sir[key].values, rtol=1e-10), f'{key} should match the model written by hand'

    with pytest.raises(ValueError, match='rho'):
        load_camdl(sir_ir, pars=dict(beta=0.4), **kw) # A parameter with no value and no bounds needs one
    return sim


def test_camdl_flat():
    """ A stratified IR is imported flat; an event, a time function, and an ODE compartment are listed """
    sc.heading('Testing camdl import of what is not translated')
    ir = sc.dcp(sir_ir)
    m = ir['model']
    m['compartments'] = [{'name': f'{c}_{g}'} for c in 'SIR' for g in ['a', 'b']]
    m['model_structure'] = {'dimensions': [{'name': 'group', 'values': ['a', 'b']}]}
    m['bindings'] = [{'name': f'N_{g}', 'expr': {'pop_sum': [f'{c}_{g}' for c in 'SIR']}} for g in 'ab']
    m['transitions'] = [
        {'name': f'infection_{g}', 'stoichiometry': [[f'S_{g}', -1], [f'I_{g}', 1]], 'rate': op('mul', op('mul', op('mul', par('beta'), {'time_func': {'name': 'season'}}), pop(f'S_{g}')), {'reduce': [op('div', pop(f'I_{h}'), {'binding_ref': f'N_{h}'}) for h in 'ab']})}
        for g in 'ab'] + [{'name': f'recovery_{g}', 'stoichiometry': [[f'I_{g}', -1], [f'R_{g}', 1]], 'rate': op('mul', par('gamma'), pop(f'I_{g}'))} for g in 'ab']
    m['time_functions'] = [{'name': 'season', 'kind': {'sinusoidal': {}}}]
    m['observations'] = []
    m['initial_conditions'] = {'S_a': {'deterministic': {'const': 990.0}}, 'S_b': {'deterministic': {'const': 1000.0}}, 'I_a': {'deterministic': {'const': 10.0}}}
    m['interventions'] = [{'name': 'seed', 'kind': 'event', 'fire': {'scheduled': {'at_times': [10.0]}}, 'actions': [{'add': {'compartment': 'I_b', 'count': {'const': 5.0}}}]}]
    with pytest.warns(ss.errors.ImportIncomplete):
        sim = load_camdl(ir, pars=dict(rho=0.5, k=1), **kw)
    skipped = [name for name,why in sim.imported.incomplete]
    assert 'season' in skipped and 'seed' in skipped, 'the time function and the event should be listed'
    sim.run()
    res = sim.results.sir
    assert res.I_b.values[30] > 1, 'the infection should reach the second group'
    assert np.allclose(sum(res[f'{c}_{g}'].values for c in 'SIR' for g in 'ab'), 2000), 'no one should be lost'

    # A guard against an empty group (N_b > 0 ? I_b/N_b : 0), which without it would be 0/0
    for tr in m['transitions'][:2]:
        terms = tr['rate']['bin_op']['right']['reduce']
        terms[1] = {'cond': {'pred': op('gt', {'binding_ref': 'N_b'}, {'const': 0.0}), 'then': terms[1], 'else': {'const': 0.0}}}
    m['initial_conditions'].pop('S_b')
    m['interventions'] = []
    with pytest.warns(ss.errors.ImportIncomplete): # The time function is still there
        sim = load_camdl(ir, pars=dict(rho=0.5, k=1), **kw)
    sim.run()
    assert np.isfinite(sim.results.sir.I_a.values).all() and sim.results.sir.R_a.values[-1] > 100, 'the empty group should not stop the epidemic'
    return sim


def export_sims():
    """ An SIR with bounds and a prior on beta, and an SEIRD with births, deaths, and a lognormal dwell time (written as Erlang stages) """
    S, I, R = ss.states('S I R')
    beta = ss.par(ss.perday(0.3), bounds=[0.1, 0.9], prior=ss.lognormal(log_mean=-1, log_std=0.5), doc='transmission rate')
    sir = ss.Sim(ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ ss.days(10), init_n=dict(I=10), name='sir'), total_pop=1e4, unit='day', stop=80, verbose=0)
    S, E, I, R, D = ss.states('S E I R D')
    seird = ss.Disease(births=ss.birth(S, rate=ss.peryear(0.02)), infection=ss.Transition(S, E, rate=ss.perday(0.4)*I), onset=ss.Transition(E, I, dur=ss.days(4)),
                       outcome=ss.Transition(I, [R, D], dur=ss.lognormal(ss.days(8), ss.days(3)), p=[None, 0.1]), deaths=ss.death([S, E, I, R], rate=ss.peryear(0.01)), init_prev=0.01, name='seird')
    return dict(sir=sir, seird=ss.Sim(seird, total_pop=1e5, unit='day', stop=120, verbose=0))


def test_camdl_export():
    """ A sim as a camdl model: its transitions as total rates, the kinds, bounds, and prior of its parameters, and their values in a scenario """
    sc.heading('Testing export to camdl')
    sims = export_sims()
    text, summary, lost = to_camdl(sims['sir'])
    assert "time_unit = 'days" in text and 'compartments { S, I, R }' in text
    assert 'beta : rate in [0.1, 0.9] ~ log_normal(mu = -1.0, sigma = 0.5)' in text and "#' transmission rate" in text
    assert 'infection : S --> I @ beta * S * I / N' in text and 'dur_I : duration' in text
    assert 'scenarios {' in text and '      beta = 0.3' in text and "to   = 80 'days" in text
    text, summary, lost = to_camdl(sims['seird'])
    assert 'births : --> S @ unchecked_dim(' in text and 'deaths_1 : S --> @ rate_deaths * S' in text # A rate of numbers alone is declared to be in people per day
    assert any('Erlang' in line for line in lost)
    return text


@pytest.mark.skipif(not has_camdl, reason='needs camdl (its environment in the lingua_franca repo; set STARSIM_LF)')
def test_camdl_export_run(tmp_path, monkeypatch):
    """ The export read back (compiled with camdlc), and run in camdl, against the sim run as an ODE with one RK4 step a day, as camdl takes """
    sc.heading('Testing export to camdl, read back and run in camdl')
    monkeypatch.setenv('CAMDLC', str(camdl_bin/'camdlc'))
    for name,sim in export_sims().items():
        file = tmp_path/f'{name}.camdl'
        sc.savetext(file, to_camdl(sim)[0])
        ref = sim.new(solver='ode', substeps=1).run().results[name]
        back = ss.import_from(file, format='camdl', **kw).run()
        res = back.results[name]
        stages = lambda r, k: sum(r[c].values for c in r.keys() if c == k or (c.startswith(f'{k}_') and c[len(k)+1:].isdigit()))
        for k in ['S', 'I', 'R']:
            assert np.allclose(ref[k].values, stages(res, k), rtol=1e-9, atol=1e-6), f'{name}: {k} should be the same read back'
        if name == 'sir':
            p = back.diseases.sir.graph.pars.beta
            assert p.bounds == [0.1, 0.9] and isinstance(p.prior, ss.lognormal) and p.prior.log_form and p.doc == 'transmission rate'

        # In camdl itself, which writes counts as whole numbers
        traj = tmp_path/f'{name}.tsv'
        subprocess.run([str(camdl_bin/'camdl'), 'simulate', str(file), '--no-progress', '--scenario', 'baseline', '--backend', 'ode', '-o', str(traj), '--output-dir', str(tmp_path/'store')], check=True, capture_output=True)
        rows = [line.rstrip('\n').split('\t') for line in sc.loadtext(traj).splitlines() if not line.startswith('#')]
        data = {c:np.array([float(r[i]) for r in rows[1:]]) for i,c in enumerate(rows[0])}
        for k in ['S', 'I', 'R']:
            got = sum(v for c,v in data.items() if c == k or (c.startswith(f'{k}_') and c[len(k)+1:].isdigit()))
            assert np.abs(ref[k].values - got).max() <= 0.5*sum(c == k or c.startswith(f'{k}_') for c in data) + 1e-6, f'{name}: {k} should be the same in camdl, to its rounding'
    return back


if __name__ == '__main__':
    T = sc.timer()
    sim1 = test_camdl_ir()
    sim2 = test_camdl_flat()
    text = test_camdl_export()
    if has_camdl:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
            test_camdl_export_run(sc.path(tmp), mp)
    T.toc()
