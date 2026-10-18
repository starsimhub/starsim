"""
Write a summer2 model to JSON, for `ss.load(..., format='summer2')`

This runs where summer2 is installed, and needs neither Starsim nor Sciris, so it can
be copied into any summer2 environment. Run it on a function that builds the model,
or a module that makes one called `model`:

    python dump_summer2.py my_model.py:build_model model.json
    python dump_summer2.py my_model.py model.json params.json # With the values of its parameters

or call `dump(model, 'model.json', pars=params)` from Python. The model is written
before it is run (it need not be run), with each flow as it was added (before
stratification), the stratifications in the order they were applied, and the initial
number in every stratified compartment. A function of time (summer2's linear and
sigmoidal interpolations) is written as its values over time.
"""
import sys
import json
import runpy
import numbers
import numpy as np

__all__ = ['dump', 'model_dict']


ops = dict(add='+', subtract='-', multiply='*', divide='/', true_divide='/', power='**')


def par_name(obj):
    """ The name of a summer2 Parameter (its key; older versions called it name) """
    return getattr(obj, 'key', None) or getattr(obj, 'name')


def series(obj, times):
    """ A function of time only (a linear or sigmoidal interpolation) as {time: value}: at its points if linear, else every tenth of a step between them """
    import jax # summer2's functions are written in JAX, and are evaluated at many times at once with vmap
    try:
        knots = [float(x) for x in obj.args[1].args[0].args] # Function(get_scale_data, [Function(_capture_array, x)])
    except (AttributeError, IndexError, TypeError, ValueError):
        knots = [times[0], times[1]]
    if getattr(obj.func, '__name__', '') == 'interpolate_linear':
        grid = np.array(knots)
    else:
        grid = np.append(np.arange(knots[0], knots[-1], times[2]/10), knots[-1])
    xs, ys = obj.args[1].evaluate(), obj.args[2].evaluate()
    vals = jax.vmap(lambda t: obj.func(t, xs, ys))(grid)
    return {float(t):float(v) for t,v in zip(grid, vals)}


def expr(obj, found=None, times=None):
    """ A computegraph object (a Function of Parameters, DerivedOutputs, data, and time) as an expression in Python syntax; functions of time are added to found as series """
    obj = getattr(obj, 'obj', obj)
    kind = type(obj).__name__
    if isinstance(obj, numbers.Number):
        return repr(float(obj))
    elif kind == 'Data':
        return repr(obj.data.tolist() if hasattr(obj.data, 'tolist') else obj.data)
    elif kind == 'Parameter':
        return par_name(obj)
    elif kind == 'DerivedOutput':
        return obj.key
    elif kind == 'ModelVariable':
        return 't' if 'time' in str(obj) else str(obj)
    elif kind == 'Function':
        name = getattr(obj.func, '__name__', str(obj.func))
        if name.startswith('interpolate_') and found is not None and times is not None: # A function of time only
            key = f'series_{len(found) + 1}'
            found[key] = series(obj, times)
            return key
        args = [expr(a, found, times) for a in obj.args]
        if name in ops and len(args) == 2:
            return f'({args[0]} {ops[name]} {args[1]})'
        return f'{name}({", ".join(args)})'
    return f'<{obj}>'


def value(param, defaults, times=None):
    """ A parameter as JSON: a number, a named parameter with its value, or the text of a function (with the values of its functions of time) """
    if param is None:
        return None
    if isinstance(param, numbers.Number):
        return float(param)
    obj = getattr(param, 'obj', param) # summer2 wraps most values in a GraphObjectParameter
    kind = type(obj).__name__
    if kind == 'Data':
        data = obj.data
        return data.tolist() if hasattr(data, 'tolist') else data
    elif kind == 'Parameter':
        return dict(par=par_name(obj), value=defaults.get(par_name(obj)))
    elif isinstance(obj, numbers.Number):
        return float(obj)
    elif hasattr(obj, 'tolist'): # A plain array, e.g. a mixing matrix
        return obj.tolist()
    found = {}
    text = expr(obj, found, times)
    return dict(function=text, series=found) if found else dict(function=text)


def adjustment(adj, defaults):
    """ A flow or infectiousness adjustment: [kind, value], or None """
    if adj is None:
        return None
    return [type(adj).__name__.lower(), value(adj.param, defaults)]


def model_dict(model, pars=None):
    """ The model as a dict that can be written as JSON, with the parameters' values (pars, else their defaults) """
    defaults = {**(model.get_default_parameters() or {}), **(pars or {})}
    missing = sorted(set(model.get_input_parameters()) - set(defaults)) if hasattr(model, 'get_input_parameters') else []
    if missing: # summer2 needs every value to work out the initial population
        errormsg = f'The model\'s named Parameters {missing} have no values: give them as pars=dict(...), or as a JSON file after the output file on the command line'
        raise ValueError(errormsg)
    times = [float(model.times[0]), float(model.times[-1]), float(model.timestep)]

    # Flows as they were added: the first copy of each, with its base parameter
    flows = {}
    for flow in model.flows:
        if flow.name in flows:
            if flow.source is not None and flow.source.name not in flows[flow.name]['sources']:
                flows[flow.name]['sources'].append(flow.source.name)
            continue
        flows[flow.name] = dict(name=flow.name, kind=type(flow).__name__, sources=[flow.source.name] if flow.source is not None else [],
                                dest=flow.dest.name if flow.dest is not None else None, param=value(flow.param, defaults, times))

    # Stratifications, in the order applied
    strats = []
    for s in model._stratifications:
        fadj = {}
        for name,items in s.flow_adjustments.items():
            fadj[name] = [dict(adjustments={k:adjustment(v, defaults) for k,v in adj.items()}, source_strata=dict(src), dest_strata=dict(dst)) for adj,src,dst in items]
        iadj = {comp:{k:adjustment(v, defaults) for k,v in adj.items()} for comp,adj in s.infectiousness_adjustments.items()}
        strats.append(dict(name=s.name, strata=list(s.strata), compartments=[c.name for c in s.compartments],
                           population_split={k:value(v, defaults) for k,v in s.population_split.items()},
                           flow_adjustments=fadj, infectiousness_adjustments=iadj,
                           mixing_matrix=value(s.mixing_matrix, defaults), ageing=s.is_ageing(), strain=s.is_strain()))

    # The initial number in each compartment, stratified
    pop = model.get_initial_population(defaults)
    initial = [dict(name=c.name, strata=dict(c.strata), value=float(v)) for c,v in zip(model.compartments, pop.values)]

    # Derived outputs, as requested
    outputs = {}
    for name,req in model._derived_output_requests.items():
        out = {}
        for k,v in req.items():
            if k == 'request_type':
                out[k] = str(getattr(v, 'value', v))
            elif k == 'func':
                out[k] = expr(v)
            elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
                out[k] = v
            else:
                out[k] = str(v)
        outputs[name] = out

    return dict(
        format = 'summer2',
        compartments = [c.name for c in model._original_compartment_names],
        infectious = [c.name for c in model._infectious_compartments],
        initial_population = {str(k):value(v, defaults) for k,v in model._init_pop_dist.items()},
        initial = initial,
        times = [float(model.times[0]), float(model.times[-1])],
        timestep = float(model.timestep),
        ref_date = str(model.ref_date) if getattr(model, 'ref_date', None) is not None else None,
        flows = list(flows.values()),
        stratifications = strats,
        outputs = outputs,
        parameters = {k:(float(v) if isinstance(v, numbers.Number) else str(v)) for k,v in defaults.items()},
    )


def dump(model, filename, pars=None):
    """ Write a summer2 model to a JSON file, with the values of its parameters (pars, else their defaults) """
    d = model_dict(model, pars=pars) # Before opening the file, so a failure leaves no empty file
    with open(filename, 'w') as f:
        json.dump(d, f, indent=1)
    return filename


if __name__ == '__main__':
    target, filename = sys.argv[1], sys.argv[2]
    pars = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    path, _, func = target.partition(':')
    ns = runpy.run_path(path, run_name='dump_summer2')
    model = ns[func]() if func else ns['model']
    dump(model, filename, pars=pars)
    print(f'Wrote {filename}')
