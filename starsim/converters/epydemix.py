"""
Import an epydemix model

epydemix (https://github.com/epistorm/epydemix) builds a model in Python, so it is read
either as a live `EpiModel` (if epydemix is installed alongside Starsim) or as the JSON
written by this file run as a script (see below), which runs in epydemix's own environment. Each
compartment becomes a state; each mediated transition exposure to its agent, through
the population's contact matrices (summed over the settings, as epydemix does); each
spontaneous transition a rate per day; and the population's age groups a
stratification. epydemix runs in daily steps of a chain binomial, so the sim is
`solver='binomial'` with `substeps=1`.

Writing the JSON: run as a script, this file runs where epydemix is installed, and
needs neither Starsim nor Sciris, so it can be copied into any epydemix environment.
Run it on a function that builds the model, or on a script that makes one called
`model`:

    python epydemix.py my_model.py:build_model model.json
    python epydemix.py my_model.py model.json

or call `dump(model, 'model.json')` from Python. The function may return the model, or
a dict of the model (as `model`) and the arguments of `simulate()` (`start_date`,
`end_date`, `initial_conditions_dict`, parameters, ...). A script is run as it is, and
the last call to `simulate()` or `run_simulations()` with the model gives the run
settings: the dates, the step, the initial numbers in each compartment and age group,
and any parameters passed to the run (such as `beta` in a calibration wrapper).

Used with: compartments; declared modules
"""
import os
import sys
if __name__ == '__main__': # Run as a script, this folder would hide epydemix itself (with this file, epydemix.py)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path = [p for p in sys.path if os.path.abspath(p or '.') != here]
import re
import ast
import json
import runpy
import numbers
import numpy as np
if __package__: # As part of Starsim; run as a script in epydemix's environment, the rest needs neither Sciris nor Starsim
    import sciris as sc
    import starsim as ss
    from .core import Source, Code, build, factors, product

__all__ = ['load_epydemix', 'dump', 'model_dict', 'record']


def read(source):
    """ The model as a dict: from a dict, a JSON file, or a live epydemix model """
    if isinstance(source, dict):
        return source
    elif hasattr(source, 'transitions_list') and hasattr(source, 'population'):
        return model_dict(source)
    return sc.loadjson(source)


def ages(groups):
    """ The lower edges of age groups named like "0-4", "5-19", and "65+", or None if they are not ages """
    edges = []
    for g in groups:
        m = re.match(r'^\s*(\d+)\s*(-\s*\d+|\+)?\s*$', str(g))
        if not m:
            return None
        edges.append(int(m.group(1)))
    if edges[0] != 0 or any(b <= a for a,b in zip(edges[:-1], edges[1:])):
        return None
    return edges


def same(a, b):
    """ Whether two expressions are the same """
    return ast.dump(a) == ast.dump(b)


def complement(node):
    """ For an expression 1 - X, X; otherwise None """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub) and isinstance(node.left, ast.Constant) and node.left.value == 1:
        return node.right
    return None


def branch(a, b):
    """ For two rates r*X and r*(1 - X) (in either order), (r, X, which has X); otherwise None """
    try:
        (na, da), (nb, db) = factors(ast.parse(a, mode='eval').body), factors(ast.parse(b, mode='eval').body)
    except SyntaxError:
        return None
    for (n1, d1, n2, d2, first) in [(na, da, nb, db, 0), (nb, db, na, da, 1)]:
        for i,f in enumerate(n2):
            x = complement(f)
            if x is None:
                continue
            match = [j for j,g in enumerate(n1) if same(g, x)]
            if not match:
                continue
            rest1 = [g for j,g in enumerate(n1) if j != match[0]]
            rest2 = [g for j,g in enumerate(n2) if j != i]
            if len(rest1) == len(rest2) and all(same(g, h) for g,h in zip(rest1, rest2)) and len(d1) == len(d2) and all(same(g, h) for g,h in zip(d1, d2)):
                return ast.unparse(product(rest1, d1)), ast.unparse(x), first
    return None


def load_epydemix(source, start=None, stop=None, verbose=None, **kwargs):
    """
    Read an epydemix model into a sim

    Args:
        source (str/dict/EpiModel): the JSON file written by `epydemix.py` (or its contents), or the model itself
        start (str): the start date (default: the recorded run's, or epydemix's default, 2020-01-01)
        stop (str): the end date, as epydemix's `end_date` (default: the recorded run's, or 2020-12-31)
        verbose (bool): whether to print how each transition was read
        kwargs (dict): passed to `ss.Sim()`

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated

    **Example**:

        # In the epydemix environment: python epydemix.py my_model.py model.json
        sim = ss.import_from('model.json', format='epydemix')
    """
    d = read(source)
    file = source if isinstance(source, str) else None
    src = Source(name='epydemix', format='epydemix', file=file)
    src.totals = ['N']
    comps = d['compartments']
    pop = d['population']
    groups = pop['groups']
    Nk = np.array(pop['Nk'], dtype=float)
    n = len(groups)
    run = d.get('run') or {}
    dt = run.get('dt', 1.0)
    start = sc.ifelse(start, run.get('start_date'), '2020-01-01')
    stop = sc.ifelse(stop, run.get('end_date'), '2020-12-31')
    dates = np.arange(np.datetime64(start, 'D'), np.datetime64(stop, 'D') + np.timedelta64(1, 'D')) if dt == 1 else None # The day of each step
    edges = ages(groups)
    dim = 'age' if edges is not None else 'group'

    # Parameters: the model's, then those passed to the run, which take their place (as in simulate())
    raw = dict(d['parameters'])
    raw.update(run.get('parameters') or {})
    overrides = d.get('overrides') or {}

    def series(values, name):
        """ Values for each step as an ss.timeseries, with a point only where the value changes """
        if dates is None:
            src.skip(f'parameter {name}', f'it varies in time, but the step is {dt} days; only daily steps are translated')
            return float(values[0])
        values = np.asarray(values, dtype=float)[:len(dates)]
        points = {str(dates[0]): float(values[0])}
        for i in range(1, len(values)):
            if values[i] != values[i-1]:
                points[str(dates[i])] = float(values[i])
        return ss.timeseries(points) if len(points) > 1 else float(values[0])

    def value(name, p):
        """ A parameter's value in Starsim, or None if it was skipped """
        kind, val = p['kind'], p['value']
        ovs = overrides.get(name)
        if ovs:
            if kind not in ['number', 'time'] or any(o['value']['kind'] != 'number' for o in ovs) or dates is None:
                src.skip(f'overrides of {name}', 'only overrides of a number by a number, in daily steps, are translated')
            else:
                vals = np.full(len(dates), float(val)) if kind == 'number' else np.resize(np.asarray(val, dtype=float), len(dates))
                for o in ovs:
                    on = (dates >= np.datetime64(o['start_date'])) & (dates <= np.datetime64(o['end_date']))
                    vals[on] = o['value']['value']
                return series(vals, name)
        if kind == 'number':
            return val
        elif kind == 'time':
            return series(val, name)
        elif kind == 'age':
            return ss.by(val, dims=[dim]) if n > 1 else val[0]
        elif kind == 'time_age':
            src.skip(f'parameter {name}', 'it varies both in time and by age group; it is taken at its first value, by group')
            return ss.by(val[0], dims=[dim]) if n > 1 else val[0][0]
        elif kind == 'expression':
            src.exprs[name] = val
            return None
        src.skip(f'parameter {name}', f'a value that is not a number or an array ({val})')
        return None

    # States, with the initial numbers by group from the recorded run
    init = run.get('initial') or {}
    for c in comps:
        vals = np.array(init.get(c, np.zeros(n)), dtype=float)
        src.add_state(c, init=None if not vals.sum() else ss.by(vals.tolist(), dims=[dim]) if n > 1 else float(vals.sum()))
    rest = comps[0]
    for c in comps: # The rest of each group is in the state that holds most of it, as Starsim fills the first state
        if c in init and np.sum(init[c]) > np.sum(init.get(rest, 0)):
            rest = c
    if rest != comps[0]:
        src.states = sc.obj({rest:src.states[rest], **{k:v for k,v in src.states.items() if k != rest}})
    src.states[rest].init = None
    people = Nk # The number in each group
    if init:
        counted = np.sum([np.array(v, dtype=float) for v in init.values()], axis=0)
        if np.abs(counted - Nk).max() >= 0.5: # Whole people
            people = counted
            src.notes.append(f'population    the initial numbers add up to {counted.sum():n}, not the population ({Nk.sum():n}), which is the sim\'s population; epydemix divides by the {Nk.sum():n} in the force of infection')
    else:
        src.skip('initial conditions', 'no run was recorded, so the initial numbers are not known; give init=')

    # Transitions
    used = set()
    flows = []
    for tr in d['transitions']:
        s, t, kind, params = tr['source'], tr['target'], tr['kind'], tr['params']
        if kind == 'mediated':
            rate, agent = params
            flows.append((s, t, f'({rate}) * {agent} / N', None, kind))
        elif kind == 'spontaneous':
            flows.append((s, t, str(params), str(params), kind))
        else:
            src.skip(f'transition {s} -> {t}', f'of a custom kind ("{kind}"), registered with register_transition_kind(), which is not translated')
            continue
        for node in ast.walk(ast.parse(str(params[0] if kind == 'mediated' else params), mode='eval')):
            if isinstance(node, ast.Name):
                used.add(node.id)

    # Two spontaneous exits from a state at r*X and r*(1 - X) are a branch, as written by hand
    done = set()
    names = set()
    for i,(s, t, rate, plain, kind) in enumerate(flows):
        if i in done:
            continue
        name = f'{s}_to_{t}'
        if name in names: # epydemix adds up transitions between the same compartments
            name = f'{name}_{kind}'
        names.add(name)
        if plain is not None:
            for j in range(i+1, len(flows)):
                s2, t2, rate2, plain2, _ = flows[j]
                if j in done or s2 != s or plain2 is None or t2 == t:
                    continue
                found = branch(plain, plain2)
                if found:
                    common, x, which = found
                    dsts = [t2, t] if which else [t, t2] # The destination with X first
                    name = f'{s}_to_{dsts[0]}_or_{dsts[1]}'
                    src.add_flow(s, dsts, common, name=name, p=[x, None])
                    src.notes.append(f'{name:16s} {s} -> {t} at {plain} and {s} -> {t2} at {plain2}: one branch at {common}, with p={x} to {dsts[0]} (its result counts both)')
                    done.update([i, j])
                    break
        if i not in done:
            src.add_flow(s, t, rate, name=name)
            done.add(i)

    for name,p in raw.items():
        if name in comps:
            continue
        val = value(name, p)
        if val is not None:
            if name not in used:
                src.notes.append(f'{name:16s} a parameter not used in any rate')
            src.add_par(name, val, doc=f'epydemix parameter "{name}"')

    # Interventions on the contacts
    for iv in d.get('interventions') or []:
        what = f'a reduction by {iv["reduction_factor"]}' if iv.get('reduction_factor') is not None else 'a new matrix'
        src.skip(f'intervention {iv.get("name") or iv["layer"]}', f'{what} of the "{iv["layer"]}" contacts from {iv["start_date"]} to {iv["end_date"]}; the contacts are one matrix, so this is by hand (e.g. ss.modify on the infection, if every setting is reduced)')

    # Groups and contacts
    matrix = np.sum([np.array(m, dtype=float) for m in pop['contacts'].values()], axis=0) if pop['contacts'] else np.ones((n, n))
    if n > 1:
        levels = edges if edges is not None else list(groups)
        src.stratify(**{dim:levels}, split=(people/people.sum()).tolist())
        src.config.networks = Code(f'ss.contacts({matrix.tolist()!r}' + (f', dims={dim!r}' if dim != 'age' else '') + ')')
        src.notes.append(f'contacts      the {len(pop["contacts"])} settings of {pop["name"]} ({sc.strjoin(pop["contacts"].keys())}) added together')

    # Run settings: epydemix's end date is the last day it steps through, so the sim stops a day later
    src.config.update(unit='day', dt=dt, start=start, stop=str(np.datetime64(stop, 'D') + np.timedelta64(1, 'D')), solver='binomial', substeps=1, total_pop=float(people.sum()))
    if edges is not None:
        src.config.use_aging = False
    return build(src, verbose=verbose, **kwargs)


#%% Writing the model to JSON, in epydemix's own environment (needs neither Sciris nor Starsim)

calls = {} # The last simulate() call for each model, by id


def jsonable(val):
    """ A parameter value as JSON: a number, a list, or a string """
    if isinstance(val, numbers.Number):
        return float(val)
    elif isinstance(val, str):
        return val
    elif hasattr(val, 'tolist'):
        return val.tolist()
    elif isinstance(val, (list, tuple)):
        return [jsonable(v) for v in val]
    return str(val)


def date(val):
    """ A date as YYYY-MM-DD """
    return None if val is None else str(np.datetime64(val, 'D'))


def parameter(val):
    """ A parameter as JSON: its kind (number, time, age, time_age) and value """
    if isinstance(val, numbers.Number):
        return dict(kind='number', value=float(val))
    elif isinstance(val, str):
        return dict(kind='expression', value=val)
    arr = np.asarray(val, dtype=float)
    if arr.ndim == 0:
        return dict(kind='number', value=float(arr))
    elif arr.ndim == 1:
        return dict(kind='time', value=arr.tolist()) # epydemix reads a 1D array as values over time
    elif arr.ndim == 2 and arr.shape[0] == 1:
        return dict(kind='age', value=arr[0].tolist())
    elif arr.ndim == 2:
        return dict(kind='time_age', value=arr.tolist())
    return dict(kind='other', value=str(val))


def run_dict(model, call):
    """ The settings of a run: the dates, the step, the initial numbers, and the parameters passed to it """
    if call is None:
        return None
    call = dict(call)
    init = call.pop('initial_conditions_dict', None)
    pct = call.pop('percentage_in_agents', 0.0005)
    if init is None:
        init = model.create_default_initial_conditions(percentage_in_agents=pct)
    out = dict(
        start_date = date(call.pop('start_date', '2020-01-01')),
        end_date   = date(call.pop('end_date', '2020-12-31')),
        dt         = float(call.pop('dt', 1.0) or 1.0),
        initial    = {k:np.asarray(v, dtype=float).tolist() for k,v in init.items()},
    )
    for key in ['Nsim', 'resample_frequency', 'resample_aggregation_compartments', 'resample_aggregation_transitions', 'fill_method', 'rng', 'epimodel', 'contact_matrices', 'simulation_dates', 'apply_linear_approximation']:
        call.pop(key, None)
    out['parameters'] = {k:parameter(v) for k,v in call.items()}
    return out


def model_dict(model, run=None):
    """ The model as a dict that can be written as JSON; run is a dict of the arguments to simulate(), if any """
    if run is None:
        run = calls.get(id(model))
    pop = model.population
    trs = []
    for tr in model.transitions_list:
        params = tr.params
        params = [jsonable(p) for p in params] if isinstance(params, (list, tuple)) else jsonable(params)
        trs.append(dict(source=tr.source, target=tr.target, kind=tr.kind, params=params))
    overrides = {k:[dict(start_date=date(o['start_date']), end_date=date(o['end_date']), value=parameter(o['value'])) for o in v] for k,v in model.overrides.items()}
    interventions = []
    for iv in model.interventions:
        interventions.append(dict(layer=iv['layer'], start_date=date(iv['start_date']), end_date=date(iv['end_date']), name=iv.get('name', ''),
                                  reduction_factor=iv.get('reduction_factor'), new_matrix=jsonable(iv['new_matrix']) if iv.get('new_matrix') is not None else None))
    d = dict(
        format        = 'epydemix',
        name          = model.name,
        compartments  = list(model.compartments),
        transitions   = trs,
        parameters    = {k:parameter(v) for k,v in model.parameters.items()},
        overrides     = overrides,
        interventions = interventions,
        population    = dict(name=pop.name, groups=[str(g) for g in pop.Nk_names], Nk=np.asarray(pop.Nk, dtype=float).tolist(),
                             contacts={k:np.asarray(v, dtype=float).tolist() for k,v in pop.contact_matrices.items()}),
        run           = run_dict(model, run),
    )
    return d


def dump(model, filename, run=None):
    """
    Write the model (and the settings of a run, if given or recorded) to a JSON file

    Args:
        model (`EpiModel`): the epydemix model, or a dict of it and the arguments of `simulate()`
        filename (str): the JSON file to write
        run (dict): the settings of the run; if None, those recorded
    """
    with open(filename, 'w') as f:
        json.dump(model_dict(model, run=run), f, indent=1)
    return filename


def record():
    """ Record the arguments of each call to simulate() (which run_simulations() makes too), by model """
    import epydemix
    import epydemix.model
    import epydemix.model.epimodel as em
    original = em.simulate
    def simulate(epimodel, *args, **kwargs):
        names = ['start_date', 'end_date', 'initial_conditions_dict', 'percentage_in_agents', 'dt']
        calls[id(epimodel)] = dict(zip(names, args), **kwargs)
        return original(epimodel, *args, **kwargs)
    for mod in [em, epydemix.model, epydemix]:
        if getattr(mod, 'simulate', None) is original:
            mod.simulate = simulate
    return


def main(target, filename):
    """ Build the model from a function ("file.py:function") or a script (with a variable "model"), and write it """
    try:
        import matplotlib
        matplotlib.use('Agg') # Scripts often plot
    except ImportError:
        pass
    record()
    path, _, func = target.partition(':')
    scope = runpy.run_path(path, run_name='__main__' if not func else '__dump__')
    run = None
    if func:
        out = scope[func]()
        if isinstance(out, dict):
            out = dict(out)
            model = out.pop('model')
            run = out or None
        else:
            model = out
    else:
        model = scope['model']
    dump(model, filename, run=run)
    print(f'Wrote {filename}: {len(model.compartments)} compartments, {len(model.transitions_list)} transitions, {len(model.parameters)} parameters', file=sys.stderr)
    return


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
