"""
Test importing camdl models, from camdl's IR, a .camdl file, and a pycamdl model
"""
import sys
import runpy
import shutil
import tempfile
import warnings
import numpy as np
import pandas as pd
import sciris as sc
import pytest
import starsim as ss
import paths
from starsim.interop.camdl import load_camdl

lfd = paths.lfd # The examples, with the originals' results, if this folder is here
camdlc = lfd / '_envs/camdl/bin/camdlc' if (lfd / '_envs/camdl/bin/camdlc').exists() else shutil.which('camdlc') or 'camdlc' # The one built for the examples; another on the PATH may be older
pycamdl = paths.camdl/'pycamdl'
golden = lfd / '_envs/camdl/src/ocaml/golden' # camdl's library of models, as compiled IR
kw = dict(verbose=False)

def op(name, left, right):
    return {'bin_op': {'op': name, 'left': left, 'right': right}}

par, pop = lambda name: {'param': name}, lambda name: {'pop': name}

# The IR camdlc writes for examples/camdl/sir/original/sir.camdl, trimmed of what the importer does not read
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
    assert pars.beta.bounds == [0.05, 1.0] and isinstance(pars.beta.prior, ss.lognorm_im), 'bounds and priors should be kept'
    assert pars.beta.doc == 'transmission rate'
    assert isinstance(sim.diseases.sir.graph.outcomes.cases.likelihood, ss.nbinom), 'the likelihood should be kept'

    # By hand, as in vignettes/camdl/sir/sir.py
    beta, gamma, rho, k = ss.par(0.4), ss.par(0.15), ss.par(0.6), ss.par(10)
    S, I, R = ss.states(3)
    infection = S >> I @ beta * I
    cases = ss.outcome(rho * infection)
    sir = ss.Disease(infection, I >> R @ gamma, cases, init=dict(I=10), name='sir')
    ref = ss.Sim(sir, total_pop=10_000, unit='day', stop=ss.weeks(12), solver='ode', substeps=1, verbose=0).run() # One RK4 step a day, as camdl takes
    for key in ['n_I', 'n_R', 'infection', 'cases']:
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
    assert res.n_I_b.values[30] > 1, 'the infection should reach the second group'
    assert np.allclose(sum(res[f'n_{c}_{g}'].values for c in 'SIR' for g in 'ab'), 2000), 'no one should be lost'

    # A guard against an empty group (N_b > 0 ? I_b/N_b : 0), which without it would be 0/0
    for tr in m['transitions'][:2]:
        terms = tr['rate']['bin_op']['right']['reduce']
        terms[1] = {'cond': {'pred': op('gt', {'binding_ref': 'N_b'}, {'const': 0.0}), 'then': terms[1], 'else': {'const': 0.0}}}
    m['initial_conditions'].pop('S_b')
    m['interventions'] = []
    with pytest.warns(ss.errors.ImportIncomplete): # The time function is still there
        sim = load_camdl(ir, pars=dict(rho=0.5, k=1), **kw)
    sim.run()
    assert np.isfinite(sim.results.sir.n_I_a.values).all() and sim.results.sir.n_R_a.values[-1] > 100, 'the empty group should not stop the epidemic'
    return sim


@pytest.mark.skipif(not lfd.exists(), reason=paths.no_lfd)
@pytest.mark.skipif(not sc.path(camdlc).exists(), reason='needs camdlc')
def test_camdl_file():
    """ A .camdl file, compiled with camdlc, against camdl's own ODE run """
    sc.heading('Testing camdl import from a .camdl file')
    folder = lfd / 'examples/camdl/sir'
    sim = load_camdl(folder / 'original/sir.camdl', pars=sir_pars, camdlc=str(camdlc), **kw).run()
    ref = pd.read_csv(folder / 'reference/results.csv')
    res = sim.results.sir
    for key in 'SIR':
        theirs = ref[ref.series == f'ode_{key}'].value.values # Written as whole numbers
        assert np.abs(res[f'n_{key}'].values - theirs).max() <= 0.5, f'{key} should agree with camdl to its rounding'
    theirs = ref[ref.series == 'ode_cases'].value.values
    weekly = res.cases.values[1:].reshape(-1, 7).sum(axis=1) # The cases of each week, as camdl reports them
    assert np.allclose(weekly, theirs, atol=1e-4), 'the weekly cases should agree with camdl (written to 5 digits)'
    return sim


@pytest.mark.skipif(not golden.exists(), reason=f'needs camdl\'s golden models: {golden}')
def test_camdl_golden():
    """ camdl's own library of models (its compiler's golden IR): each imports and runs, and what is not translated is listed """
    sc.heading('Testing camdl\'s golden models')
    for path in sorted(golden.glob('*.ir.json')):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # Most list something in ss.errors.ImportIncomplete
            sim = load_camdl(str(path), **kw)
            sim.pars.verbose = 0
            if path.name == 'phenom_mixing_unchecked.ir.json': # camdl's example of a count used for a share, with its unit check turned off: far too fast for the step, so the run stops rather than give numbers that are not
                with pytest.raises(ValueError, match='not a number'):
                    sim.run()
                continue
            sim.run()
        assert sim.complete, f'{path.name} should run'
        if path.name == 'polio_spatial_5.ir.json': # Its SIAs are off unless the with_sia scenario enables them: listed, not dropped
            listed = dict(sim.imported.incomplete)
            assert 'sia_north' in listed and 'with_sia' in listed['sia_north']
    return sim


@pytest.mark.skipif(not lfd.exists(), reason=paths.no_lfd)
@pytest.mark.skipif(not pycamdl.exists(), reason='needs pycamdl: set STARSIM_CAMDL to where the camdl repo is')
def test_pycamdl(tmp_path):
    """ A pycamdl model (SEIR by age and patch, with aging and movement), written with dump_pycamdl.py, against pycamdl's own ODE run """
    sc.heading('Testing pycamdl import')
    sys.path.insert(0, str(pycamdl))
    from starsim.interop import dump_pycamdl
    folder = lfd / 'examples/camdl/stratified'
    model = runpy.run_path(str(folder / 'original/stratified.py'), run_name='original')['m']
    path = tmp_path / 'model.json'
    dump_pycamdl.dump(model, path)
    sim = load_camdl(str(path), **kw).run()
    res = sim.results[sim.diseases[0].name]
    ref = pd.read_csv(folder / 'reference/results.csv')
    for c in 'SEIR':
        theirs = ref[ref.series == f'ode_novax_{c}'].value.values
        ours = sum(res[k].values for k in res.keys() if k.startswith(f'n_{c}_'))
        assert np.allclose(ours, theirs, rtol=1e-9), f'{c} should agree with pycamdl'
    return sim


if __name__ == '__main__':
    T = sc.timer()
    sim1 = test_camdl_ir()
    sim2 = test_camdl_flat()
    sim3 = test_camdl_file()
    sim5 = test_camdl_golden()
    sim4 = test_pycamdl(sc.path(tempfile.mkdtemp()))
    T.toc()
