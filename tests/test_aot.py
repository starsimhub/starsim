"""AOT dispatch and artifact checks.

To include native integration tests, first build an artifact and set
STARSIM_AOT_TEST_ARTIFACT to its directory. The tests otherwise run only the
fast routing/CLI checks; building a native artifact takes a few minutes.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import starsim as ss


def make_sim(solver='ode'):
    S, I, R = ss.states(3)
    disease = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init=dict(I=0.1), name='sir')
    sim = ss.Sim(disease, total_pop=100, unit='day', stop=10, solver=solver, verbose=0)
    sim.jax = False
    return sim


@pytest.fixture
def clean_aot():
    saved = ss.kernels.native
    ss.aot.unload()
    try:
        yield
    finally:
        ss.kernels.native = saved


@pytest.fixture
def artifact():
    folder = os.getenv('STARSIM_AOT_TEST_ARTIFACT')
    if not folder:
        pytest.skip('Set STARSIM_AOT_TEST_ARTIFACT to a built AOT artifact')
    return Path(folder).resolve()


def test_routing(clean_aot, monkeypatch):
    """AOT does not imply other entries are compiled, or override plain mode."""
    run = object()
    monkeypatch.setattr(ss.kernels, 'native', SimpleNamespace(run=run))
    monkeypatch.setattr(ss.kernels, 'is_compiled', lambda: False)
    monkeypatch.setattr(ss.kernels, 'chosen', False)
    monkeypatch.setattr(ss.kernels, 'plain_total', 0.)
    owner = SimpleNamespace(compiled=False, program=None)
    choose = ss.solvers.compartmental.choose_kernels
    with ss.options.context(compile='auto'):
        assert ss.kernels.choose(entry='run').run is run
        assert choose(owner, 1, lambda: 0., entry='run').run is run
        assert owner.compiled is False
        assert ss.kernels.choose() is ss.kernels.uncompiled()
    with ss.options.context(compile='never'):
        assert ss.kernels.choose(entry='run') is ss.kernels.uncompiled()
        assert choose(owner, 1, lambda: 0., entry='run') is ss.kernels.uncompiled()


def test_cli(monkeypatch, tmp_path):
    from starsim.internals import cli
    build = ss.aot.build
    calls = []
    monkeypatch.setattr(ss.aot, 'build', lambda folder, cpu: calls.append((folder, cpu)))
    monkeypatch.setattr(ss, 'compile', lambda: calls.append('jit'))
    cli.main(['compile', '--aot', str(tmp_path)])
    cli.main(['compile', '--aot', str(tmp_path), '--cpu', 'native'])
    cli.main(['compile'])
    assert calls == [(str(tmp_path), 'generic'), (str(tmp_path), 'native'), 'jit']
    with pytest.raises(ValueError, match='cpu'):
        build(tmp_path, cpu='invalid')


def test_lifecycle(artifact, clean_aot):
    originals = ss.kernels.run, ss.kernels.evaluate
    with ss.options.context(compile='auto'):
        entries = ss.aot.load(artifact)
        assert ss.options.compile == 'auto'
        assert (ss.kernels.run, ss.kernels.evaluate) == originals
        with ss.options.context(compile='never'):
            make_sim().run()
        assert entries['run'].hits == entries['run'].fallbacks == 0
        make_sim().run()
        assert entries['run'].hits == 1
        assert entries['run'].fallbacks == 0
        ss.aot.unload()
        assert ss.kernels.native is None
        assert ss.options.compile == 'auto'


@pytest.mark.parametrize('solver', ['ode', 'binomial', 'tau', 'gillespie'])
def test_results(artifact, clean_aot, solver):
    # This reference has no native extension or JIT dependency.
    with ss.options.context(compile='never'):
        reference = make_sim(solver).run()
    entries = ss.aot.load(artifact)
    with ss.options.context(compile='auto'):
        actual = make_sim(solver).run()
    assert entries['run'].hits == 1
    for key, value in reference.results.flatten().items():
        np.testing.assert_allclose(np.asarray(actual.results.flatten()[key]), np.asarray(value), rtol=1e-12, atol=1e-12)


def test_crossed_diseases(artifact, clean_aot):
    def make():
        from test_kernels import make_sir
        sir = make_sir()
        S, I = ss.states('S I')
        other = ss.Disease(S >> I @ ss.perday(0.2)*I, I >> S @ ss.days(7), init=dict(I=0.02), name='other')
        same = ss.modify(sir.rel_sus, multiply=1.0, where=other.I, name='same')
        return ss.Sim([sir, other], connectors=same, solver='ode', total_pop=1000, unit='day', stop=20, verbose=0)
    with ss.options.context(compile='never'):
        reference = make().run()
    entries = ss.aot.load(artifact)
    with ss.options.context(compile='auto'):
        actual = make().run()
    assert actual.diseases.sir.engine.cross is not None
    assert entries['run'].hits == 1
    for key, value in reference.results.flatten().items():
        np.testing.assert_allclose(np.asarray(actual.results.flatten()[key]), np.asarray(value), rtol=1e-12, atol=1e-12)


def test_evaluate_semantics(artifact, clean_aot):
    """Keep NumPy division semantics and safely dispatch a different array layout."""
    entries = ss.aot.load(artifact)
    k = ss.kernels
    ints = lambda x: np.array(x, dtype=np.int64)
    tape = k.Tape(ints([k.CONST, k.CONST, k.DIV]), ints([0, 1, 0]), ints([-1, -1, 1]),
                  np.array([True, True]), ints([0, 1]), ints([0, 1]), ints([1]), ints([0, 1]),
                  2, True, ints([0, 1]), False)
    y = np.array([[9., 18.], [1., 2.]])
    constants = np.array([[1., 0.], [0., 0.]])
    out, grouped = np.zeros((3, 2)), np.zeros(3, dtype=bool)
    entries['evaluate'](y, tape, constants, out, grouped)
    assert np.isposinf(out[2, 0]) and out[2, 1] == 0 # Starsim defines 0/0 as zero for empty groups
    assert entries['evaluate'].hits == 1
    expected = out.copy()
    out.fill(0)
    entries['evaluate'](np.asfortranarray(y), tape, constants, out, grouped)
    assert entries['evaluate'].fallbacks == 1
    np.testing.assert_array_equal(out, expected)


def test_reject_stale_artifact(artifact, clean_aot, tmp_path):
    ss.aot.load(artifact)
    active = ss.kernels.native
    manifest = json.loads((artifact/'manifest.json').read_text())
    manifest['environment']['sources']['kernels.py'] = 'stale'
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='does not match'):
        ss.aot.load(tmp_path)
    assert ss.kernels.native is active


@pytest.mark.parametrize('model', ['sir', 'seir', 'ages', 'towns', 'commute', 'malaria', 'delay'])
@pytest.mark.parametrize('solver', ['ode', 'binomial', 'tau', 'gillespie'])
def test_model_matrix(artifact, clean_aot, model, solver):
    """Native and JIT agree across mixing, movement, staged delays, and reservoirs."""
    import sciris as sc
    import test_kernels as kernels
    make, pars = kernels.models[model]
    pars = sc.mergedicts(pars, total_pop=2000, stop=20, solver=solver, rand_seed=7, verbose=0)
    with ss.options.context(compile='always'):
        reference = ss.Sim(make(), **pars)
        reference.jax = False
        reference.run()
        entries = ss.aot.load(artifact)
        actual = ss.Sim(make(), **pars)
        actual.jax = False
        actual.run()
    assert entries['run'].hits == 1
    assert entries['run'].fallbacks == 0
    assert kernels.largest_diff(reference, actual) < 1e-12


def test_fresh_process(artifact, tmp_path):
    """Auto mode uses native code without JIT, even with an empty Numba cache."""
    code = '''
import sys
import starsim as ss
from numba.core import event
class Compiles(event.Listener):
    def on_start(self, ev):
        raise AssertionError('Unexpected compilation: ' + ev.data['dispatcher'].py_func.__name__)
    def on_end(self, ev):
        pass
with event.install_listener('numba:compile', Compiles()):
    entries = ss.aot.load(sys.argv[1])
    from test_aot import make_sim
    for solver in ['ode', 'binomial', 'tau', 'gillespie']:
        make_sim(solver).run()
    assert entries['run'].hits == 4
    assert entries['run'].fallbacks == 0
    assert ss.options.compile == 'auto'
'''
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path/'numba'), STARSIM_COMPILE='auto',
               STARSIM_UNCOMPILED_WARNING='0', STARSIM_INSTALL_FONTS='0',
               PYTHONPATH=os.pathsep.join([str(Path(ss.__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]))
    subprocess.run([sys.executable, '-c', code, str(artifact)], env=env, check=True, capture_output=True, text=True)
