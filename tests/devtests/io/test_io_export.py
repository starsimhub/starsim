"""
Test exporting a sim to each format (ss.export_to()): each export of a small SIR is read back with ss.import_from() and gives the same run; and where the framework is installed (its environment in the lingua_franca repo's `_envs/`; set STARSIM_LF to say where the repo is), it reads or runs the export. Also runs as a hub submission (ss.converters.to_hub())
"""
import os
import csv
import glob
import json
import subprocess
import warnings
import xml.etree.ElementTree as ET
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import corpus
import paths

kw = dict(verbose=0)
converters = sc.path(ss.converters.__file__).parent


def framework(env, exe):
    """ A program in a framework's environment, skipping the rest of the test if it is not there """
    path = paths.lf/'_envs'/env/'bin'/exe
    if not path.exists():
        pytest.skip(f'needs {env} (its environment in the lingua_franca repo; set STARSIM_LF)')
    return path


def run(*args, **kwargs):
    """ Run a program, check that it worked, and return what it printed """
    out = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=300, **kwargs)
    assert out.returncode == 0, out.stderr
    return out.stdout


def export(fmt, tmp_path, ext, sim=None):
    """ Export a sim (by default, an SIR in days) to a file; returns the sim and the file """
    sim = sim if sim is not None else corpus.io_sir()
    filename = tmp_path/f'sir.{ext}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ss.errors.LossyExport)
        ss.export_to(sim, fmt, filename=filename, verbose=False)
    return sim, filename


def same(sim, back, I='I'):
    """ The sim and the export read back give the same epidemic (summing a state written as stages, I, which the export may name otherwise) """
    a = sim.run().results.sir
    b = back.run().results[back.diseases[0].name]
    n = len(a.I)
    assert np.allclose(corpus.stages(b, I)[:n], a.I.values, rtol=1e-9, atol=1e-6), 'The export read back should give the same epidemic'
    return a


def test_json(tmp_path):
    sc.heading('Testing the JSON form: a stratified model through a file, and a file runs no code')
    S, I, R = ss.states('S I R')
    gamma = ss.par(0.1, bounds=[0.01, 1.0])
    R0 = ss.par(ss.by(patch=dict(urban=2.5, rural=1.8)))
    sir = ss.Disease(S >> I @ R0*gamma*I, I >> R @ gamma, init_n=dict(I=ss.by(patch=dict(urban=10, rural=1))), name='sir')
    sir.stratify(patch=['urban', 'rural'])
    sim, filename = export('json', tmp_path, 'json', sim=ss.Sim(sir, total_pop=2000, unit='day', stop=100, **kw))
    back = ss.import_from(filename, **kw) # The format is told from the file
    a = same(sim, back)
    assert np.allclose(a.I.strata, back.results.sir.I.strata)
    model = sc.loadjson(filename)
    model['diseases'][0]['transitions']['recovery'] = "I >> R @ __import__('os').getcwd()"
    with pytest.raises(ValueError):
        ss.import_from(model, format='json')


def test_sbml(tmp_path):
    sc.heading('Testing export to SBML, run in libRoadRunner')
    sim, filename = export('sbml', tmp_path, 'xml')
    ET.parse(filename)
    a = same(sim, ss.import_from(filename, stop=60, **kw))
    python = framework('sbml', 'python')
    out = run(python, '-c', f'import roadrunner; r = roadrunner.RoadRunner({str(filename)!r}); print(r.simulate(0, 60, 61)["[I]"].tolist())')
    assert np.allclose(json.loads(out.splitlines()[-1]), a.I.values, rtol=1e-4)


def test_odin(tmp_path):
    sc.heading('Testing export to odin, run in odin')
    sim, filename = export('odin', tmp_path, 'R')
    a = same(sim, ss.import_from(filename, format='odin', stop=60, **kw))

    # A lognormal dwell time is written as Erlang stages, and said to be (the same in every exporter)
    seird = corpus.seird()
    with pytest.warns(ss.errors.LossyExport, match='7 exponential stages'):
        ss.export_to(seird, 'odin', filename=tmp_path/'seird.R')
    back = ss.import_from(tmp_path/'seird.R', unit='day', stop=120, **kw).run()
    assert np.allclose(corpus.stages(back.results.seird, 'I'), seird.run().results.seird.I.values, rtol=1e-9)
    rscript = framework('odin', 'Rscript')
    out = run(rscript, '-e', f'gen <- odin::odin("{filename}", verbose=FALSE); y <- gen$new()$run(0:60); cat(y[, "I"], sep=",")')
    assert np.allclose(np.array(out.split(','), dtype=float), a.I.values, rtol=1e-4)


def test_siminf(tmp_path):
    sc.heading('Testing export to SimInf, run in SimInf')
    sim, filename = export('siminf', tmp_path, 'R')
    same(sim, ss.import_from(filename, format='siminf', solver='ode', **kw))
    rscript = framework('siminf', 'Rscript')
    out = run(rscript, '-e', f'pdf(NULL); set.seed(1); source("{filename}"); cat(sum(trajectory(result)$R))')
    assert float(out) > 0


def test_epimodel(tmp_path):
    sc.heading('Testing export to EpiModel, read back by dump_epimodel.R')
    sim, filename = export('epimodel', tmp_path, 'R')
    assert 'new.mod = sir' in sc.loadtext(filename)
    rscript = framework('epimodel', 'Rscript')
    run(rscript, converters/'dump_epimodel.R', filename, tmp_path/'sir.json') # Runs the model in EpiModel, and records it
    same(sim, ss.import_from(tmp_path/'sir.json', format='epimodel', **kw))


@pytest.mark.parametrize('fmt', ['epipack', 'epydemix', 'epymorph', 'summer2', 'memilio'])
def test_python(tmp_path, fmt):
    sc.heading(f'Testing export to {fmt}, built in {fmt} and read back')
    sim, script = export(fmt, tmp_path, 'py')
    compile(sc.loadtext(script), script, 'exec')
    python = framework(fmt, 'python')
    run(python, converters/f'{fmt}.py', script, tmp_path/'sir.json', cwd=tmp_path) # Runs the script in the framework, and records the model it builds
    back = ss.import_from(tmp_path/'sir.json', format=fmt, **kw)
    same(sim, back.new(solver='ode'), I='Infected' if fmt == 'memilio' else 'I') # As an ODE, whatever solver the framework's own runs are read as


def test_camdl(tmp_path, monkeypatch):
    sc.heading('Testing export to camdl, compiled and run in camdl')
    sim, filename = export('camdl', tmp_path, 'camdl')
    sim = sim.new(solver=ss.solvers.ode(substeps=1)) # One RK4 step a day, as camdl takes
    assert 'infection : S --> I @ beta * S * I / N' in sc.loadtext(filename)
    camdlc = framework('camdl', 'camdlc')
    monkeypatch.setenv('CAMDLC', str(camdlc))
    a = same(sim, ss.import_from(filename, format='camdl', **kw))
    traj = tmp_path/'sir.tsv'
    run(camdlc.parent/'camdl', 'simulate', filename, '--no-progress', '--scenario', 'baseline', '--backend', 'ode', '-o', traj, '--output-dir', tmp_path/'store')
    rows = list(csv.DictReader((line for line in open(traj) if not line.startswith('#')), delimiter='\t'))
    assert np.abs(np.array([float(r['I']) for r in rows]) - a.I.values).max() <= 0.5 # camdl writes whole numbers


def test_emulsion(tmp_path):
    sc.heading('Testing export to EMULSION, run in EMULSION')
    sim, filename = export('emulsion', tmp_path, 'yaml')
    back = ss.import_from(filename, **kw)
    assert isinstance(back.init().solver, ss.solvers.binomial) # As EMULSION runs it
    same(sim, back.new(solver='ode'))
    emulsion = framework('emulsion', 'emulsion')
    run(emulsion, 'run', filename, '--runs', 1, '--seed', 1, '--quiet', '--output-dir', tmp_path/'out', cwd=tmp_path)
    rows = list(csv.DictReader(open(tmp_path/'out'/'counts.csv')))
    assert max(float(r['I']) for r in rows) > 10


def test_flepimop(tmp_path):
    sc.heading('Testing export to flepiMoP, run in flepiMoP')
    sim, filename = export('flepimop', tmp_path, 'yml')
    sim = sim.new(solver=ss.solvers.ode(substeps=1)) # One RK4 step a day, as flepiMoP takes
    a = same(sim, ss.import_from(filename, format='flepimop', **kw))
    bin = framework('flepimop', 'flepimop').parent
    env = dict(os.environ, PATH=f'{bin}{os.pathsep}{os.environ["PATH"]}', NUMBA_NUM_THREADS='1')
    run(bin/'flepimop', 'simulate', filename.name, '--write-csv', '--no-write-parquet', '--jobs', 1, '--out-id', 'test', cwd=tmp_path, env=env)
    out = pd.read_csv(glob.glob(str(tmp_path/'model_output'/'*'/'test'/'seir'/'*.seir.csv'))[0])
    I = out[(out.mc_value_type == 'prevalence') & (out.mc_name == 'I')]['all'].values
    assert np.allclose(I, a.I.values[:len(I)], rtol=0, atol=0.05)


def test_atomica(tmp_path):
    sc.heading('Testing export to Atomica, run in Atomica')
    sim = corpus.io_sir(unit='year', start=2000, stop=2000.25, dt=ss.days(1), solver=ss.solvers.ode(integrator='euler', substeps=1)) # Stepped as Atomica steps
    sim, filename = export('atomica', tmp_path, 'xlsx', sim=sim)
    a = same(sim, ss.import_from(filename, format='atomica', start=2000, stop=2000.25, dt=1/365, **kw))
    python = framework('atomica', 'python')
    code = 'import json, logging, numpy as np, atomica as at; at.logger.setLevel(logging.WARNING); ' \
           f'F = at.ProjectFramework({str(filename)!r}); D = at.ProjectData.new(F, np.arange(2000, 2001), pops={{"pop": "Population"}}, transfers=0); ' \
           'P = at.Project(framework=F, databook=D.to_spreadsheet(), do_run=False); P.settings.update_time_vector(start=2000, end=2000.25, dt=1/365); ' \
           'pop = P.run_sim().model.pops[0]; print(json.dumps({c.name: c.vals.tolist() for c in pop.comps}))'
    comps = json.loads(run(python, '-c', code).splitlines()[-1])
    assert np.allclose(comps['S'][:len(a.S)], a.S.values, rtol=1e-9), 'Atomica should step the model as Starsim\'s Euler method does'


def test_versions():
    sc.heading('Testing the versions the converters were tested with')
    assert set(ss.converters.tested) == set(ss.converters.importers) | set(ss.converters.exporters), 'Every format should say which version it was tested with'
    assert f'(tested with odin {ss.converters.tested['odin'][1]})' in ss.export_to(corpus.io_sir(), 'odin', verbose=False)


def test_hub():
    sc.heading('Testing runs of two scenarios as a hub submission')
    def make(beta, label):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=S >> I @ ss.perday(0.4*beta)*I, recovery=I >> R @ 0.2, init_n=dict(I=10), name='sir')
        return ss.Sim(sir, total_pop=1e5, start='2020-02-01', stop='2020-05-01', unit='day', label=label, **kw)
    msim = ss.parallel(make(1.0, 'base'), make(0.5, 'lockdown'), n_runs=2, workers=1)
    hub = ss.converters.to_hub(msim, targets=dict(infection=['inc inf']), origin_date='2020-03-01', output_type='sample', location='US')
    assert list(hub.columns) == ['origin_date', 'scenario_id', 'location', 'target', 'horizon', 'output_type', 'output_type_id', 'value']
    for scenario,group in hub.groupby('scenario_id'): # Each scenario's runs are its own: the samples match its runs' weekly totals
        weekly = msim.results.sir.infection.select(scenario=scenario).values[:, 30:30+56].reshape(2, 8, 7).sum(axis=2) # The 8 whole weeks after 1 March (index 29)
        assert np.allclose(group.pivot(index='output_type_id', columns='horizon', values='value').values, weekly)
