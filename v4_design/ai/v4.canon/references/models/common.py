"""
Shared harness for the canon review: a tight scipy reference for the book's equations, and scoring of a v4 sim against it

Each chapter file defines models as `name: dict(build=fn returning an unrun ss.Sim, ref=fn returning dict of reference series on the sim's time grid, map=dict of v4 result -> ref key)`.
"""
import io
import time
import inspect
import contextlib
import numpy as np
import sciris as sc
import starsim as ss
from scipy.integrate import solve_ivp

PY = '/software/conda/envs/v4/bin/python'


def ode(f, y0, t, **kw):
    """ Solve dy/dt = f(t, y) tightly and return the solution at times t, as an array (n_vars, n_t) """
    sol = solve_ivp(f, (t[0], t[-1]), np.asarray(y0, dtype=float), t_eval=t, method='DOP853', rtol=1e-12, atol=1e-14, **kw)
    if not sol.success:
        sol = solve_ivp(f, (t[0], t[-1]), np.asarray(y0, dtype=float), t_eval=t, method='LSODA', rtol=1e-11, atol=1e-12, **kw)
    assert sol.success, sol.message
    return sol.y


def series(sim, key):
    """ A result by dotted key, e.g. 'sir.n_I', as a float array; 'sir.n_I|patch=b' selects one group (added for chapters 7 and 8) """
    sel = {}
    if '|' in key:
        key, spec = key.split('|', 1)
        for item in spec.split(','):
            k, v = item.split('=')
            sel[k] = v
    obj = sim.results
    for k in key.split('.'):
        obj = obj[k]
    if sel:
        obj = obj.select(**sel)
    arr = np.asarray(obj.values if hasattr(obj, 'values') else obj, dtype=float)
    if arr.ndim > 1: # A grouped result with no selection: the total over groups
        arr = arr.reshape(arr.shape[0], -1).sum(axis=1)
    return arr


def rel_err(a, b):
    """ Max abs difference, relative to the max of the reference """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    scale = max(np.abs(b).max(), 1e-300)
    return float(np.abs(a[:n] - b[:n]).max() / scale)


def capture(fn):
    """ Run fn, returning (value, printed text, error) """
    buf = io.StringIO()
    err = None
    val = None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            val = fn()
        except Exception as E:
            err = f'{type(E).__name__}: {E}'
    return val, buf.getvalue(), err


def score(name, model, others=('binomial', 'gillespie', 'abm'), n_agents=10_000, verbose=True):
    """ Score one model: build and run it under auto, compare with the reference, then try the other solvers """
    out = sc.objdict(name=name, book=model.get('book', ''), shows=model.get('shows', ''))
    out.lines = len([l for l in inspect.getsource(model['build']).splitlines()[1:] if l.strip() and not l.strip().startswith(('#', '"""', 'return'))])
    T = sc.timer()
    sim, text, err = capture(lambda: model['build']().run())
    out.time = T.tocout(output=True) if hasattr(T, 'tocout') else None
    out.error = err
    out.notes = [l.strip() for l in text.splitlines() if 'note' in l.lower() or 'warn' in l.lower()]
    if err:
        out.solver = None
        out.agree = None
    else:
        out.solver = sim.pars.solver if not hasattr(sim, 'solver') else sim.solver
        try:
            ref = model['ref'](sim)
            errs = model['check'](sim, ref) if 'check' in model else {k: rel_err(series(sim, k), ref[v]) for k, v in model['map'].items()} # 'check' (chapters 7 and 8): a custom comparison, e.g. of peak and final size for a stochastic run
        except Exception as E:
            errs = {}
            out.error = f'scoring: {type(E).__name__}: {str(E)[:200]}'
        out.errs = errs
        out.agree = max(errs.values()) if errs else None
        tol = model.get('tol', 1e-6)
        out.passed = out.agree is not None and out.agree <= tol
        if out.agree is not None and not out.passed and model.get('fine', True): # Separate the model from the integrator: rerun with many substeps
            simf, _, ef = capture(lambda: model['build'](substeps=64, verbose=0).run())
            if ef is None:
                out.agree_fine = max(rel_err(series(simf, k)/model.get('scale', 1), ref[v]) for k, v in model['map'].items())
    out.others = {}
    if err is None and others:
        base_key = list(model['map'].keys())[0]
        base = series(sim, base_key)
        for solver in others:
            kw = dict(solver=solver, verbose=0)
            if solver == 'abm':
                kw['n_agents'] = n_agents
            s2, _, e2 = capture(lambda: model['build'](**kw).run())
            if e2:
                out.others[solver] = f'error: {e2[:200]}'
            else:
                y = series(s2, base_key)
                out.others[solver] = f'peak {y.max():.4g} vs {base.max():.4g}; final {y[-1]:.4g} vs {base[-1]:.4g}'
    if verbose:
        fine = f" fine={out.agree_fine:.2e}" if 'agree_fine' in out else ''
        print(f"{name:28s} solver={out.solver} agree={out.agree if out.agree is None else f'{out.agree:.2e}'}{fine} err={out.error}")
        for k, v in out.others.items():
            print(f'    {k:10s} {v}')
        for n in out.notes:
            print(f'    note: {n[:200]}')
    return out


def run_all(models, select=None, **kw):
    results = []
    for name, m in models.items():
        if select and name not in select:
            continue
        results.append(score(name, m, **kw))
    return results
