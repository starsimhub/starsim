"""
Write an epydemix model to JSON, for `ss.load(..., format='epydemix')`

This runs where epydemix is installed, and needs neither Starsim nor Sciris, so it can
be copied into any epydemix environment. Run it on a function that builds the model,
or on a script that makes one called `model`:

    python dump_epydemix.py my_model.py:build_model model.json
    python dump_epydemix.py my_model.py model.json

or call `dump(model, 'model.json')` from Python. The function may return the model, or
a dict of the model (as `model`) and the arguments of `simulate()` (`start_date`,
`end_date`, `initial_conditions_dict`, parameters, ...). A script is run as it is, and
the last call to `simulate()` or `run_simulations()` with the model gives the run
settings: the dates, the step, the initial numbers in each compartment and age group,
and any parameters passed to the run (such as `beta` in a calibration wrapper).
"""
import sys
import json
import runpy
import numbers
import numpy as np

__all__ = ['dump', 'model_dict', 'record']

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
    """ Write the model (and the settings of a run, if given or recorded) to a JSON file """
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
