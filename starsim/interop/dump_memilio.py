"""
Write a MEmilio model to JSON, for `ss.interop.load(..., format='memilio')`

This runs where MEmilio's Python bindings (`memilio-simulation`) are installed, and
needs neither Starsim nor Sciris, so it can be copied into any MEmilio environment.
MEmilio's ODE models are fixed in C++, and a script only sets their parameters, so
this runs the script as it is and records the model it simulates: the first call to
`simulate()` or `simulate_flows()`, the first `Simulation` or `MobilitySimulation`
that is advanced, or the first `ParameterStudy` or `GraphParameterStudy` that is run.
Run it on a script, or on a function in one:

    python dump_memilio.py ode_secir_simple.py model.json
    python dump_memilio.py my_model.py:run_simulation model.json

or call `dump(model, 'model.json', t0=0, tmax=100)` from Python. What is written is
the model type, the number of age groups, every parameter by age group (with its
distribution, if it has one), the initial populations, the contact matrices with
their dampings (and the matrix the model uses at the start and either side of each
damping), and the run's times. For a graph, it is the first node's model, plus each
node's model and each edge's mobility coefficients (`graph`); for a parameter study,
also the number of runs (`study`).
"""
import sys
import json
import runpy
import numbers
import importlib
import numpy as np

__all__ = ['dump', 'model_dict', 'graph_dict', 'record']

models = ['osir', 'oseir', 'osecir', 'osecirvvs', 'oseirdb', 'omseirs4', 'oseir_metapop', 'ssir', 'ssirs']
calls = [] # The (model or graph, t0, tmax, dt, function, runs) of each run, in order


def model_type(model):
    """ The name of the model's module, e.g. 'osecir' """
    return type(model).__module__.split('_simulation_')[-1]


def distribution(val):
    """ The distribution of an UncertainValue, for a parameter study, as JSON (None if it has none) """
    dist = val.get_distribution() if hasattr(val, 'get_distribution') else None
    if dist is None:
        return None
    elif hasattr(dist, 'lower_bound'):
        return dict(type='uniform', lower=float(dist.lower_bound), upper=float(dist.upper_bound))
    elif hasattr(dist, 'standard_dev'): # Its bounds cannot be read from Python
        return dict(type='normal', mean=float(dist.mean), std=float(dist.standard_dev))
    return dict(type=type(dist).__name__)


def value(val):
    """ A parameter as JSON: its kind and value, and its distribution if it has one """
    if isinstance(val, numbers.Number):
        return dict(kind='number', value=float(val))
    elif hasattr(val, 'value') and isinstance(getattr(val, 'value'), numbers.Number): # An UncertainValue
        out = dict(kind='number', value=float(val.value))
        dist = distribution(val)
        if dist:
            out['distribution'] = dist
        return out
    return None


def groups(arr, n):
    """ An array by age group (AgeGroupArray), as JSON: its values, and their distributions if any have one """
    from memilio.simulation import AgeGroup
    vals = [arr[AgeGroup(i)] for i in range(n)]
    out = dict(kind='age', value=[float(v.value) if hasattr(v, 'value') else float(v) for v in vals])
    dists = [distribution(v) for v in vals]
    if any(dists):
        out['distributions'] = dists
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
    sampled = [dict(value=value(d.value), time=float(d.time), level=int(d.level), type=int(d.type), matrices=[int(i) for i in d.matrix_indices], weights=np.asarray(d.group_weights, dtype=float).tolist())
               for d in getattr(cp, 'dampings', []) or []] # The dampings that a parameter study draws, and which replace the matrices' own there
    if sampled:
        out['sampled_dampings'] = sampled
    holidays = len(getattr(cp, 'school_holidays', []) or [])
    if holidays:
        out['school_holidays'] = holidays
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
        try:
            val = getattr(pars, name)
        except TypeError: # A type the bindings do not convert, e.g. osecirvvs's InfectiousnessNewVariant
            other[name] = 'a parameter that MEmilio\'s Python bindings cannot read'
            continue
        if callable(val) and not hasattr(val, 'value') and not hasattr(val, 'numel'):
            continue
        out = value(val)
        if out is None and hasattr(val, 'numel'): # By age group
            try:
                out = groups(val, n)
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
    cells = {s:[model.populations[AgeGroup(i), getattr(mod.InfectionState, s)] for i in range(n)] for s in states}
    populations = {s:[float(v.value) for v in vals] for s,vals in cells.items()}
    drawn = {s:[distribution(v) for v in vals] for s,vals in cells.items()}
    drawn = {s:dists for s,dists in drawn.items() if any(dists)} # Initial numbers drawn in a parameter study
    d = dict(
        format      = 'memilio',
        version     = version(),
        model       = kind,
        groups      = n,
        states      = states,
        parameters  = parameters,
        other       = other,
        populations = populations,
        population_distributions = drawn or None,
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


def node_model(node):
    """ The model of a graph's node: the node's property in a ModelGraph, and its simulation's model in a MobilityGraph """
    prop = node.property
    return prop if hasattr(prop, 'populations') else prop.model


def graph_dict(graph, t0=0.0, tmax=100.0, dt=0.5, function='MobilitySimulation'):
    """ A graph (MobilityGraph or ModelGraph) as a dict that can be written as JSON: the first node's model, plus each node's model and each edge's mobility coefficients """
    models = [node_model(graph.get_node(i)) for i in range(graph.num_nodes)]
    d = model_dict(models[0], t0=t0, tmax=tmax, dt=dt, function=function)
    keep = ['parameters', 'other', 'populations', 'population_distributions', 'contacts']
    nodes = []
    for i,model in enumerate(models):
        md = model_dict(model, t0=t0, tmax=tmax, dt=dt)
        nodes.append(dict(id=int(graph.get_node(i).id), **{k:md[k] for k in keep}))
    edges = []
    for i in range(graph.num_edges):
        edge = graph.get_edge(i)
        prop = edge.property
        coeffs = (prop.parameters if hasattr(prop, 'parameters') else prop).coefficients # A Mobility in a MobilityGraph, MobilityParameters in a ModelGraph
        damped = sum(len(list(coeffs[k].get_dampings())) for k in range(coeffs.num_matrices))
        edges.append(dict(start=int(edge.start_node_idx), end=int(edge.end_node_idx), coefficients=np.asarray(coeffs.get_matrix_at(t0), dtype=float).ravel().tolist(), dampings=damped))
    d['graph'] = dict(nodes=nodes, edges=edges)
    return d


def dump(model, filename, runs=None, **kwargs):
    """ Write the model or graph (and its run's times, as kwargs t0, tmax, dt; and the number of runs of a parameter study) to a JSON file """
    is_graph = hasattr(model, 'num_nodes')
    d = graph_dict(model, **kwargs) if is_graph else model_dict(model, **kwargs)
    return write(d, filename, runs=runs)


def write(d, filename, runs=None):
    """ Write a model's dict to a JSON file """
    if runs is not None:
        d['study'] = dict(runs=int(runs))
    text = json.dumps(d, indent=None if 'graph' in d else 1) # A graph's edges are many short lists
    with open(filename, 'w') as f:
        f.write(text)
    return filename


def snapshot(graph, t0, dt, function):
    """ A graph as a dict, taken before a simulation or study takes it over (MobilitySimulation moves the graph, leaving the Python object empty); the end time is set when it is known """
    return graph_dict(graph, t0=t0, tmax=float('inf'), dt=dt, function=function)


def finish(d, tmax):
    """ Set the end time of a snapshot, and drop the contact matrices sampled after it """
    d['run']['tmax'] = float(tmax)
    for c in [d['contacts']] + [node['contacts'] for node in d['graph']['nodes']]:
        if c:
            c['samples'] = [s for s in c['samples'] if s['time'] <= tmax]
    return d


def record():
    """ Record the model and times of each run: through simulate(), simulate_flows(), a Simulation or MobilitySimulation that is advanced, or a ParameterStudy or GraphParameterStudy that is run """
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
        for cname in ['Simulation', 'FlowSimulation', 'MobilitySimulation']:
            cls = getattr(mod, cname, None)
            if cls is None:
                continue
            class Recorded(cls):
                def __init__(self, model, t0=0, dt=0.1 if cname != 'MobilitySimulation' else 1.0, *args, _cname=cname, **kwargs): # MobilitySimulation's model is its graph
                    self._run = dict(model=model, t0=t0, tmax=None, dt=dt, function=_cname)
                    if hasattr(model, 'num_nodes'):
                        self._run['graph'] = snapshot(model, t0, dt, _cname)
                    super().__init__(model, t0, dt, *args, **kwargs)
                def advance(self, tmax):
                    if self._run not in calls:
                        calls.append(self._run)
                    self._run['tmax'] = tmax
                    return super().advance(tmax)
            Recorded.__name__ = cname
            setattr(mod, cname, Recorded)
        for cname in ['ParameterStudy', 'GraphParameterStudy']:
            cls = getattr(mod, cname, None)
            if cls is None:
                continue
            class Study(cls):
                def __init__(self, model, t0, tmax, dt, num_runs, _cname=cname):
                    self._run = dict(model=model, t0=t0, tmax=tmax, dt=dt, function=_cname, runs=num_runs)
                    if hasattr(model, 'num_nodes'):
                        self._run['graph'] = snapshot(model, t0, dt, _cname)
                    super().__init__(model, t0, tmax, dt, num_runs)
                def run(self, *args, **kwargs):
                    if self._run not in calls:
                        calls.append(self._run)
                    return super().run(*args, **kwargs)
            Study.__name__ = cname
            setattr(mod, cname, Study)
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
    if 'graph' in run:
        d = finish(run['graph'], run['tmax'])
        write(d, filename, runs=run.get('runs'))
        what = f'a graph of {len(d["graph"]["nodes"])} {d["model"]} models and {len(d["graph"]["edges"])} edges'
    else:
        dump(run['model'], filename, runs=run.get('runs'), t0=run['t0'], tmax=run['tmax'], dt=run['dt'], function=run['function'])
        what = f'the {model_type(run["model"])} model'
    extra = f' (and {len(calls) - 1} later run(s), not written)' if len(calls) > 1 else ''
    study = f', a study of {run["runs"]} runs' if run.get('runs') else ''
    print(f'Wrote {filename}: {what}, from t={run["t0"]} to {run["tmax"]}{study}{extra}', file=sys.stderr)
    return


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
