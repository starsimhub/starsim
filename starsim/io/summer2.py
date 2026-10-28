"""
Import a summer2 model

summer2 (https://github.com/monash-emu/summer2) builds a model in Python, so it is read
either as a live `CompartmentalModel` (if summer2 is installed alongside Starsim) or as
the JSON written by this file run as a script (see below), which runs in summer2's own environment. Each
flow becomes one transition with the matching kind of rate, each `stratify_with()` a
stratification (with its population split, flow adjustments, infectiousness
adjustments, and mixing matrix), and the derived outputs can be worked out from the
results with `ss.io.summer2.outputs()`.

Writing the JSON: run as a script, this file runs where summer2 is installed, and
needs neither Starsim nor Sciris, so it can be copied into any summer2 environment.
Run it on a function that builds the model, or a module that makes one called `model`:

    python summer2.py my_model.py:build_model model.json
    python summer2.py my_model.py model.json params.json # With the values of its parameters

or call `dump(model, 'model.json', pars=params)` from Python. The model is written
before it is run (it need not be run), with each flow as it was added (before
stratification), the stratifications in the order they were applied, and the initial
number in every stratified compartment. A function of time (summer2's linear and
sigmoidal interpolations) is written as its values over time.
"""
import os
import sys
if __name__ == '__main__': # Run as a script, this folder would hide summer2 itself (with this file, summer2.py)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path = [p for p in sys.path if os.path.abspath(p or '.') != here]
import ast
import json
import runpy
import numbers
import numpy as np
if __package__: # As part of Starsim; run as a script in summer2's environment, the rest needs neither Sciris nor Starsim
    import sciris as sc
    import starsim as ss
    from .core import Source, Code, build

__all__ = ['load_summer2', 'outputs', 'dump', 'model_dict']

skipped_kinds = dict(ReplacementBirthFlow='births that replace the deaths in each step, which have no rate of their own')


def read(source, pars=None):
    """ The model as a dict: from a dict, a JSON file, or a live summer2 model (with the values of its parameters) """
    if isinstance(source, dict):
        return source
    elif hasattr(source, 'compartments') and hasattr(source, 'flows') and hasattr(source, '_stratifications'):
        return model_dict(source, pars=pars)
    return sc.loadjson(source)


def function_rate(src, name, param, values):
    """ A flow's parameter that is a function of named parameters and of time (as series), as an expression; None if it uses anything else """
    text = param['function']
    found = param.get('series', {})
    for i,(key,vals) in enumerate(found.items()):
        sname = f'{name}_series' if len(found) == 1 else f'{name}_series_{i+1}'
        src.add_par(sname, ss.timeseries({float(t):v for t,v in vals.items()}, mode='linear'), doc=f'summer2 function of time in flow "{name}"')
        text = text.replace(key, sname)
    try:
        used = {n.id for n in ast.walk(ast.parse(text, mode='eval')) if isinstance(n, ast.Name)}
    except SyntaxError:
        return None
    for var in used - set(src.pars):
        val = values.get(var)
        if not sc.isnumber(val):
            return None
        src.add_par(var, float(val), doc='summer2 Parameter')
    return text


def multiplier(adj, base):
    """ A summer2 adjustment as a multiplier: Multiply(x) is x, Overwrite(x) is x/base, and None is 1 """
    if adj is None:
        return 1.0
    kind, val = adj
    if not sc.isnumber(val):
        return None
    if kind == 'multiply':
        return float(val)
    elif kind == 'overwrite' and sc.isnumber(base) and base:
        return float(val)/base
    return None


def load_summer2(source, pars=None, verbose=None):
    """
    Read a summer2 model into a sim

    Args:
        source (str/dict/CompartmentalModel): the JSON file written by `summer2.py` (or its contents), or the model itself
        pars (dict): the values of the model's named parameters, instead of their defaults (summer2 is given them when it is run)
        verbose (bool): whether to print how each flow was read

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated

    **Example**:

        # In the summer2 environment: python summer2.py my_model.py:build_model model.json
        sim = ss.import_from('model.json', format='summer2')
    """
    d = read(source, pars=pars)
    values = {**d.get('parameters', {}), **(pars or {})}
    file = source if isinstance(source, str) else None
    src = Source(name='summer2', format='summer2', file=file)
    comps = d['compartments']
    infectious = d['infectious']
    src.totals = ['N']

    # States, with the initial numbers as added (the stratified numbers are used below)
    for c in comps:
        init = d['initial_population'].get(c, 0)
        init = init['value'] if isinstance(init, dict) and 'value' in init else init
        src.add_state(c, init=init if sc.isnumber(init) else 0)

    # Flows: one transition each, with its parameter
    nflows = {}
    names = {} # The name of each flow's transition, or of each one for a flow from several compartments
    base = {} # The value of each flow's parameter
    kinds = {}
    n_infection = sum(f['kind'].startswith('Infection') for f in d['flows'])
    for f in d['flows']:
        name, kind, param = f['name'], f['kind'], f['param']
        kinds[name] = kind
        if name.startswith('ageing_'): # Added by an age stratification, which ages people itself
            continue
        if kind in skipped_kinds:
            src.skip(f'flow {name}', f'a {kind}: {skipped_kinds[kind]}')
            continue
        suffix = dict(InfectionFrequencyFlow='contact_rate', InfectionDensityFlow='contact_rate', ImportFlow='number').get(kind, 'rate')
        pname = suffix if suffix == 'contact_rate' and n_infection == 1 else f'{name}_{suffix}'
        if isinstance(param, dict) and 'par' in param:
            pname, param = param['par'], values.get(param['par'], param['value'])
        if isinstance(param, dict) and 'function' in param: # A function of named parameters and time
            rate = function_rate(src, name, param, values)
            if rate is None:
                src.skip(f'flow {name}', f'its parameter is a function ({param["function"]}) of something other than named parameters and time')
                continue
            base[name] = None
        elif not sc.isnumber(param):
            src.skip(f'flow {name}', f'its parameter is a function ({param}); a value that varies in time could be an ss.timeseries')
            continue
        else:
            rate = pname
            base[name] = param
            src.add_par(pname, param, doc=f'summer2 {kind} "{name}"')
        sources = f['sources']
        sick = ' + '.join(infectious)
        if kind == 'InfectionFrequencyFlow':
            flows = [(sources[0], f['dest'], f'{rate} * ({sick}) / N', 'person')]
        elif kind == 'InfectionDensityFlow':
            flows = [(sources[0], f['dest'], f'{rate} * ({sick})', 'person')]
        elif kind in ['TransitionFlow', 'DeathFlow']:
            flows = [(s, f['dest'], rate, 'person') for s in sources]
        elif kind == 'AbsoluteFlow':
            flows = [(sources[0], f['dest'], rate, 'total')]
        elif kind == 'CrudeBirthFlow':
            flows = [(None, f['dest'], f'{rate} * N', 'total')]
        elif kind == 'ImportFlow':
            flows = [(None, f['dest'], rate, 'total')]
        else:
            src.skip(f'flow {name}', f'a {kind}, which is not translated')
            continue
        names[name] = [name] if len(flows) == 1 else [f'{name}_{s}' for s in sources] # One transition for each compartment it leaves
        for (s, dst, rate, per),tname in zip(flows, names[name]):
            src.add_flow(s, dst, rate, per=per, name=tname)
            nflows[tname] = src.flows[-1]

    # Stratifications
    mixing = {}
    unit = None
    for strat in d['stratifications']:
        sname, strata = strat['name'], strat['strata']
        n = len(strata)
        full = sorted(strat['compartments']) == sorted(comps)
        kw = sc.objdict()
        if strat['ageing']:
            kw[sname] = [float(s) for s in strata]
            kw.kind = 'age'
            unit = 'year' # summer2 ages people by the width of each age group, in years
        else:
            kw[sname] = list(strata)
            if strat['strain']:
                kw.kind = 'strain'
        if not full:
            kw.states = list(strat['compartments'])
        split = [strat['population_split'][s] for s in strata]
        kw.split = split
        adjust = sc.objdict()

        # Flow adjustments
        for fname,items in strat['flow_adjustments'].items():
            if fname not in names:
                continue
            f = next(f for f in d['flows'] if f['name'] == fname)
            into = f['dest'] in strat['compartments'] and not any(s in strat['compartments'] for s in f['sources']) # Into the stratified states, from outside them
            conds = [it for it in items if it['source_strata'] or it['dest_strata']]
            if not into:
                if conds:
                    src.skip(f'adjustment of {fname} by {sname}', f'it differs by {sc.strjoin(sorted({k for it in conds for k in [*it["source_strata"], *it["dest_strata"]]}))}, but a multiplier on a transition is by one dimension at a time')
                    continue
                mults = [multiplier(items[-1]['adjustments'][s], base[fname]) for s in strata]
                if None in mults:
                    src.skip(f'adjustment of {fname} by {sname}', f'an adjustment that is not a number ({items[-1]["adjustments"]})')
                    continue
                for tname in names[fname]:
                    adjust[tname] = mults
                continue

            # Into the stratified states: the shares of the flow are a split by level, and their sum multiplies the flow
            dims = sorted({k for it in items for k in it['source_strata']})
            bad = [it for it in items if it['dest_strata']] or len(dims) > 1
            if bad:
                src.skip(f'adjustment of {fname} by {sname}', 'it depends on the destination\'s other strata, or on more than one of the source\'s')
                continue
            if not dims: # The same shares for everyone
                mults = np.array([multiplier(items[-1]['adjustments'][s], base[fname]) for s in strata], dtype=float)
                kw.split = list(mults/mults.sum())
                if not np.isclose(mults.sum(), 1):
                    nflows[fname].rate = f'({nflows[fname].rate}) * {mults.sum():n}'
            else: # Shares by the level of an earlier stratification
                dim = dims[0]
                prev = next(s for s in d['stratifications'] if s['name'] == dim)
                table = np.ones((len(prev['strata']), n))/n # Where no adjustment matches, summer2 splits the flow equally
                for it in items:
                    for i,level in enumerate(prev['strata']):
                        if it['source_strata'].get(dim) == level:
                            table[i] = [multiplier(it['adjustments'][s], base[fname]) for s in strata]
                totals = np.round(table.sum(axis=1), 12)
                kw.split = ss.by((table/totals[:, None]).tolist(), dims=[dim, sname])
                earlier = next(k for k in src.strats if dim in k)
                old = earlier.get('adjust') or {}
                earlier['adjust'] = {**old, fname:list(np.array(old.get(fname, np.ones(len(totals))))*totals)}

        # Infectiousness
        iadj = strat['infectiousness_adjustments']
        if iadj:
            vals = {c:[multiplier(a.get(s), 1) for s in strata] for c,a in iadj.items()}
            uniq = {tuple(v) for c,v in vals.items() if c in infectious}
            if len(uniq) == 1 and None not in next(iter(uniq)):
                adjust['rel_trans'] = list(next(iter(uniq)))
            else:
                src.skip(f'infectiousness by {sname}', f'differs between the infectious compartments ({vals}), but rel_trans is one multiplier for each level')

        # Mixing
        mm = strat['mixing_matrix']
        if mm is not None:
            if isinstance(mm, list):
                mixing[sname] = mm
                if any(k == 'InfectionDensityFlow' for k in kinds.values()):
                    src.skip(f'mixing by {sname}', 'a mixing matrix with density-dependent infection is used as if the infection were frequency-dependent')
            else:
                src.skip(f'mixing by {sname}', f'a mixing matrix that is a function ({mm}), which is not translated')
        if adjust:
            kw.adjust = adjust
        src.stratify(**kw)

    # Initial numbers in the stratified compartments, where the splits would not give them
    strats = d['stratifications']
    byinit = {}
    for c in comps[1:]: # The first state takes the rest of each group
        rows = [r for r in d['initial'] if r['name'] == c]
        total = sum(r['value'] for r in rows)
        if not total:
            continue
        dims = [s for s in strats if c in s['compartments']]
        if not dims:
            continue
        arr = np.zeros([len(s['strata']) for s in dims])
        for r in rows:
            arr[tuple(s['strata'].index(r['strata'][s['name']]) for s in dims)] = r['value']
        expected = total
        for kw,s in zip([k for k,s in zip(src.strats, strats) if c in s['compartments']], dims):
            sp = kw['split']
            expected = np.multiply.outer(expected, np.array(sp)) if not isinstance(sp, ss.by) else None
            if expected is None:
                break
        if expected is None or not np.allclose(expected, arr):
            byinit[c] = ss.by(arr.tolist(), dims=[s['name'] for s in dims])

    src.states[comps[0]].init = None # The first compartment is the rest of the population, as total_pop is everyone

    # Run settings
    t0, t1 = d['times']
    src.config.update(dt=d['timestep'], stop=t1, solver='ode', total_pop=sum(r['value'] for r in d['initial']))
    if t0:
        src.config.start = t0
    if unit:
        src.config.unit = unit
    if mixing:
        src.config.networks = Code('ss.contacts(dict(' + ', '.join(f'{k}={np.asarray(v).tolist()!r}' for k,v in mixing.items()) + '))') # A matrix for each stratification, whose product is the mixing
    outs = d.get('outputs') or {}
    for c,val in byinit.items():
        src.states[c].init = val
    sim = build(src, verbose=verbose)
    sim.imported.outputs = outs
    if outs:
        sim.imported.choices.append(f'  {len(outs)} derived outputs: worked out from the results by ss.io.summer2.outputs(sim)')
    return sim


def arithmetic(text, get):
    """ Evaluate arithmetic of named outputs (e.g. "(a / b)"), allowing nothing else """
    import ast
    def ev(node):
        if isinstance(node, ast.Constant) and sc.isnumber(node.value):
            return node.value
        elif isinstance(node, ast.Name):
            return get(node.id)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        elif isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            ops = {ast.Add:np.add, ast.Sub:np.subtract, ast.Mult:np.multiply, ast.Div:np.divide, ast.Pow:np.power}
            if type(node.op) in ops:
                return ops[type(node.op)](a, b)
        errormsg = f'Cannot work out the summer2 output "{text}": only arithmetic of other outputs is translated'
        raise NotImplementedError(errormsg)
    return ev(ast.parse(text, mode='eval').body)


def outputs(sim):
    """
    summer2's derived outputs, worked out from the results of a sim imported from summer2

    As in summer2, a flow is a rate (the number moved in a step, divided by the step), a
    cumulative output is the sum of those rates, and a function output is evaluated from
    the other outputs. summer2's flow output at a time is the mean of the rates at the
    start and end of the step before it (the trapezoid rule), where this is the number that
    actually moved, so the two differ where the flow curves sharply within a step (e.g. 5%
    early in a yearly TB model's scale-up, 0.2% later).

    Args:
        sim (Sim): a sim made by `load_summer2()`, that has been run

    Returns:
        A dict of arrays, one for each derived output
    """
    reqs = sim.imported.outputs
    dis = sim.diseases[0]
    res = sim.results[dis.name]
    dt = sim.t.dt if sc.isnumber(sim.t.dt) else float(sim.t.dt.value)
    out = sc.objdict()
    def select(result, strata):
        return (result.select(**strata) if strata else result).values
    def get(name):
        if name in out:
            return out[name]
        req = reqs[name]
        kind = req['request_type']
        if kind == 'flow':
            flow = req['flow_name']
            tnames = [k for k in dis.transitions.keys() if k == flow or k.startswith(f'{flow}_')]
            val = sum(select(res[t], {**req.get('source_strata', {}), **req.get('dest_strata', {})}) for t in tnames)/dt
        elif kind == 'comp':
            val = sum(select(res[c], req.get('strata') or {}) for c in req['compartments'])
        elif kind == 'agg':
            val = sum(get(s) for s in req['sources'])
        elif kind == 'cum':
            val = np.cumsum(get(req['source']))
        elif kind == 'param_func':
            val = arithmetic(req['func'], get)
        else:
            errormsg = f'summer2 output "{name}" is of a kind ({kind}) that is not translated'
            raise NotImplementedError(errormsg)
        out[name] = np.array(val, dtype=float)
        return out[name]
    for name in reqs:
        get(name)
    return out


#%% Writing the model to JSON, in summer2's own environment (needs neither Sciris nor Starsim)

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
    ns = runpy.run_path(path, run_name='__dump__')
    model = ns[func]() if func else ns['model']
    dump(model, filename, pars=pars)
    print(f'Wrote {filename}')
