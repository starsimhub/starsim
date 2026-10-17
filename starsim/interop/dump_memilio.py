"""
Write a MEmilio model to JSON, for `ss.load(..., format='memilio')`

This runs where MEmilio's Python bindings (`memilio-simulation`) are installed, and
needs neither Starsim nor Sciris, so it can be copied into any MEmilio environment.
MEmilio's ODE models are fixed in C++, and a script only sets their parameters, so
this runs the script as it is and records the model it simulates: the first call to
`simulate()` or `simulate_flows()`, or the first `Simulation` that is advanced. Run
it on a script, or on a function in one:

    python dump_memilio.py ode_secir_simple.py model.json
    python dump_memilio.py my_model.py:run_simulation model.json

or call `dump(model, 'model.json', t0=0, tmax=100)` from Python. What is written is
the model type, the number of age groups, every parameter by age group, the initial
populations, the contact matrices with their dampings (and the matrix the model uses
at the start and either side of each damping), and the run's times.
"""
import sys
import json
import runpy
import numbers
import importlib
import numpy as np

__all__ = ['dump', 'model_dict', 'record']

models = ['osir', 'oseir', 'osecir', 'osecirvvs', 'oseirdb', 'omseirs4', 'oseir_metapop', 'ssir', 'ssirs']
calls = [] # The (model, t0, tmax, dt, function) of each run, in order


def model_type(model):
    """ The name of the model's module, e.g. 'osecir' """
    return type(model).__module__.split('_simulation_')[-1]


def value(val):
    """ A parameter as JSON: its kind and value """
    if isinstance(val, numbers.Number):
        return dict(kind='number', value=float(val))
    elif hasattr(val, 'value') and isinstance(getattr(val, 'value'), numbers.Number): # An UncertainValue
        return dict(kind='number', value=float(val.value))
    return None


def groups(arr, n):
    """ An array by age group (AgeGroupArray), as a list """
    from memilio.simulation import AgeGroup
    out = []
    for i in range(n):
        v = arr[AgeGroup(i)]
        out.append(float(v.value) if hasattr(v, 'value') else float(v))
    return out


def contacts(cp, t0, tmax):
    """ The contact matrices, their dampings, and the matrix the model uses at the start and either side of each damping """
    group = cp.cont_freq_mat
    mats = []
    times = {float(t0)}
    for k in range(group.num_matrices):
        m = group[k]
        damps = [dict(coeffs=np.asarray(d.coeffs, dtype=float).tolist(), time=float(d.time), level=int(d.level), type=int(d.type)) for d in m.get_dampings()]
        for d in damps: # MEmilio smooths each damping over the day before its time
            times.update([d['time'] - 1, d['time']])
        mats.append(dict(baseline=np.asarray(m.baseline, dtype=float).tolist(), minimum=np.asarray(m.minimum, dtype=float).tolist(), dampings=damps))
    times = sorted(t for t in times if t0 <= t <= tmax)
    samples = [dict(time=t, matrix=np.asarray(group.get_matrix_at(t), dtype=float).tolist()) for t in times]
    out = dict(matrices=mats, samples=samples)
    extra = dict(dynamic_dampings=len(getattr(cp, 'dampings', []) or []), school_holidays=len(getattr(cp, 'school_holidays', []) or []))
    out.update({k:v for k,v in extra.items() if v})
    return out


def model_dict(model, t0=0.0, tmax=100.0, dt=0.1, function='simulate'):
    """ The model as a dict that can be written as JSON """
    kind = model_type(model)
    mod = importlib.import_module(f'memilio.simulation.{kind}')
    states = [name for name,s in sorted(mod.InfectionState.__members__.items(), key=lambda item: int(item[1])) if name != 'Count']
    pars = model.parameters
    n = int(pars.get_num_groups()) if hasattr(pars, 'get_num_groups') else int(model.populations.size_AgeGroup().get()) if hasattr(model.populations, 'size_AgeGroup') else 1
    parameters = {}
    other = {}
    for name in dir(pars):
        if name.startswith('_') or name in ['ContactPatterns', 'apply_constraints', 'check_constraints', 'get_num_groups']:
            continue
        val = getattr(pars, name)
        if callable(val) and not hasattr(val, 'value') and not hasattr(val, 'numel'):
            continue
        out = value(val)
        if out is None and hasattr(val, 'numel'): # By age group
            try:
                out = dict(kind='age', value=groups(val, n))
            except Exception:
                out = None
        if out is None:
            if name == 'DynamicNPIsInfectedSymptoms': # Empty unless thresholds are set
                if len(list(val.threshold)):
                    other[name] = 'dynamic NPIs with thresholds'
            else:
                other[name] = type(val).__name__
        else:
            parameters[name] = out
    from memilio.simulation import AgeGroup
    populations = {s:[float(model.populations[AgeGroup(i), getattr(mod.InfectionState, s)].value) for i in range(n)] for s in states}
    d = dict(
        format      = 'memilio',
        version     = version(),
        model       = kind,
        groups      = n,
        states      = states,
        parameters  = parameters,
        other       = other,
        populations = populations,
        contacts    = contacts(pars.ContactPatterns, t0, tmax) if hasattr(pars, 'ContactPatterns') else None,
        run         = dict(t0=float(t0), tmax=float(tmax), dt=float(dt), function=function),
    )
    return d


def version():
    """ The version of memilio-simulation """
    try:
        import importlib.metadata as im
        return im.version('memilio-simulation')
    except Exception:
        return None


def dump(model, filename, **kwargs):
    """ Write the model (and its run's times, as kwargs t0, tmax, dt) to a JSON file """
    with open(filename, 'w') as f:
        json.dump(model_dict(model, **kwargs), f, indent=1)
    return filename


def record():
    """ Record the model and times of each run, through simulate(), simulate_flows(), or a Simulation that is advanced """
    for kind in models:
        try:
            mod = importlib.import_module(f'memilio.simulation.{kind}')
        except ImportError:
            continue
        for fname in ['simulate', 'simulate_flows']:
            func = getattr(mod, fname, None)
            if func is None:
                continue
            def wrapper(t0, tmax, dt, model, *args, _func=func, _name=fname, **kwargs):
                calls.append(dict(model=model, t0=t0, tmax=tmax, dt=dt, function=_name))
                return _func(t0, tmax, dt, model, *args, **kwargs)
            setattr(mod, fname, wrapper)
        for cname in ['Simulation', 'FlowSimulation']:
            cls = getattr(mod, cname, None)
            if cls is None:
                continue
            class Recorded(cls):
                def __init__(self, model, t0=0, dt=0.1, *args, _cname=cname, **kwargs):
                    super().__init__(model, t0, dt, *args, **kwargs)
                    self._run = dict(model=model, t0=t0, tmax=None, dt=dt, function=_cname)
                def advance(self, tmax):
                    if self._run not in calls:
                        calls.append(self._run)
                    self._run['tmax'] = tmax
                    return super().advance(tmax)
            Recorded.__name__ = cname
            setattr(mod, cname, Recorded)
    return


def main(target, filename):
    """ Run a script (or a function in it), and write the first model it simulates """
    try:
        import matplotlib
        matplotlib.use('Agg') # Scripts often plot
        import matplotlib.pyplot as plt
        plt.show = lambda *args, **kwargs: None
    except ImportError:
        pass
    record()
    path, _, func = target.partition(':')
    sys.argv = [path] # The script may parse its own arguments
    scope = runpy.run_path(path, run_name='__main__' if not func else '__dump__')
    if func:
        scope[func]()
    if not calls:
        print(f'{target} ran no MEmilio simulation, so there is no model to write', file=sys.stderr)
        sys.exit(1)
    run = calls[0]
    dump(run['model'], filename, t0=run['t0'], tmax=run['tmax'], dt=run['dt'], function=run['function'])
    extra = f' (and {len(calls) - 1} later run(s), not written)' if len(calls) > 1 else ''
    print(f'Wrote {filename}: the {model_type(run["model"])} model, from t={run["t0"]} to {run["tmax"]}{extra}', file=sys.stderr)
    return


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
