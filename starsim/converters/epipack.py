"""
Import an epipack model

epipack (https://github.com/benmaier/epipack) builds a model in Python as a list of
reaction processes, so it is read either as a live model (if epipack is installed
alongside Starsim) or as the JSON written by this file run as a script (see below), which
runs in epipack's own environment. Each compartment becomes a state and each event a
flow. An event that changes several compartments, as a fission does (A -> B + C), becomes
one flow for each pair of a compartment that loses and one that gains, plus an inflow or
outflow for the rest, at the same rate.

epipack divides its quadratic terms by a fixed population, `initial_population_size`,
unless `correct_for_dynamical_population_size` is set. Where that is the number of people
in a model that keeps them all, it is the same as dividing by the total, and the terms are
read as exposure. Otherwise the division is kept as a number, and the terms are mass action.

A `StochasticEpiModel` runs on a network or well mixed. Well mixed, each link is
to a random person, so a transmission at rate r over k links is r*k*I/N. On a network, the
mean degree is used, and the network is listed as not translated.

Writing the JSON: run as a script, this file runs where epipack is installed, and needs
neither Starsim nor Sciris, so it can be copied into any epipack environment. Run it on a
function that builds the model, or on a script that makes one called `model`:

    python epipack.py my_model.py:build_model model.json
    python epipack.py my_model.py model.json

or call `dump(model, 'model.json')` from Python. A script is run as it is, and its last
call to `integrate()` or `simulate()` gives the solver and the end time.

Used with: compartments; declared modules
"""
import os
import sys
if __name__ == '__main__': # Run as a script, this folder would hide epipack itself (with this file, epipack.py)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path = [p for p in sys.path if os.path.abspath(p or '.') != here]
import re
import json
import runpy
import numpy as np
if __package__: # As part of Starsim; run as a script in epipack's environment, the rest needs neither Sciris nor Starsim
    import sciris as sc
    from .core import Source, build

__all__ = ['load_epipack', 'dump', 'model_dict', 'record']

default_stop = 100 # epipack's models don't say how long they run


def read(source):
    """ The model as a dict: from a dict, a JSON file, or a live epipack model """
    if isinstance(source, dict):
        return source
    elif hasattr(source, 'compartments') and hasattr(source, 'set_processes') or hasattr(source, 'node_transition_events'):
        return model_dict(source)
    return sc.loadjson(source)


def split(changes):
    """ What an event does: pairs of a compartment that loses one and one that gains one, then the rest of either as outflows or inflows """
    lost = [c for c,n in changes.items() if n < 0 for _ in range(int(-n))]
    gained = [c for c,n in changes.items() if n > 0 for _ in range(int(n))]
    pairs = list(zip(lost, gained))
    pairs += [(c, None) for c in lost[len(pairs):]]
    pairs += [(None, c) for c in gained[len(lost):]]
    return pairs


def load_epipack(source, stop=None, solver=None, pars=None, verbose=None, **kwargs):
    """
    Read an epipack model into a sim

    Args:
        source (str/dict/EpiModel): the JSON file written by `epipack.py` (or its contents), or the model itself
        stop (float): the time to stop, in epipack's unit, taken to be days (default: the recorded run's, or 100)
        solver (str): the solver (default: 'gillespie' if the recorded run was `simulate()`, or else 'ode')
        pars (dict): the values of parameters that a symbolic model leaves without one
        verbose (bool): whether to print how each event was read
        kwargs (dict): passed to `ss.Sim()`

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated

    **Example**:

        # In the epipack environment: python epipack.py my_model.py model.json
        sim = ss.import_from('model.json', format='epipack')
    """
    d = read(source)
    src = Source(name='epipack', format='epipack', file=source if isinstance(source, str) else None)
    src.incomplete = [tuple(item) for item in d.get('incomplete', [])]
    comps = d['compartments']
    run = d.get('run') or {}
    stop = sc.ifelse(stop, run.get('tmax'), default_stop)
    size = float(d['population'])
    pars = sc.mergedicts(d.get('parameters'), pars)

    # Initial numbers: the rest of the population is in the state that holds most of it, as Starsim fills the first state
    init = d.get('initial')
    if init is None:
        src.skip('initial conditions', f'the model has none, so all {size:n} people start in {comps[0]}; give init= for others')
        init = {comps[0]: size}
    people = float(sum(init.values()))
    rest = max(comps, key=lambda c: init.get(c, 0))
    for c in [rest] + [c for c in comps if c != rest]:
        src.add_state(c, init=init.get(c) or None)
    src.states[rest].init = None

    # What the quadratic terms are divided by
    kept = all(sum(e['changes'].values()) == 0 for e in d['events']) # Whether the model keeps its people
    if d.get('dynamic'):
        src.totals = ['N']
        below = '/N'
    elif kept and np.isclose(size, people):
        src.totals = ['N']
        below = '/N'
    elif size == 1:
        below = ''
        src.notes.append(f'population    epipack\'s population size is 1, so its quadratic terms are products of the numbers in the states, which are not divided by anything')
    else:
        below = f'/{size:n}'
        src.notes.append(f'population    epipack divides its quadratic terms by {size:n} throughout, which is not the number of people as the model runs ({people:n} at the start)')

    # Flows
    for i,e in enumerate(d['events']):
        rate = e['rate']
        if e['kind'] == 'quadratic':
            rate = f'({rate}){below}'
        pairs = split(e['changes'])
        if len(pairs) > 1:
            src.notes.append(f'event {i:<7d} {e["changes"]}: {len(pairs)} flows at the same rate, which are independent when stochastic')
        for s,t in pairs:
            names = [f'{s}_to_{t}' if s and t else f'{s}_out' if s else f'{t}_in']
            name = names[0] if names[0] not in [f.name for f in src.flows] else f'{names[0]}_{i}'
            src.add_flow(s, t, rate, per='total', name=name)

    # Parameters
    for name,val in pars.items():
        if val is None:
            src.skip(f'parameter {name}', 'a symbol with no value, so the flows that use it are not translated; give pars=')
        else:
            src.add_par(name, float(val), doc=f'epipack parameter "{name}"')
    if any(re.search(r'\bt\b', e['rate']) for e in d['events']):
        src.skip('time', 'a rate uses the time, t; a value that changes in time could be an ss.timeseries')

    solver = sc.ifelse(solver, run.get('solver'), 'ode')
    src.config.update(unit='day', dt=1, start=0, stop=stop, solver=solver, total_pop=people, use_aging=False)
    return build(src, verbose=verbose, **kwargs)


#%% Writing the model to JSON, in epipack's own environment (needs neither Sciris nor Starsim)

calls = {} # The last run of each model, by id


def clean(name):
    """ A compartment or parameter (a string, a tuple, or a symbol) as a name in Python """
    out = re.sub(r'\W+', '_', str(name) if not isinstance(name, tuple) else '_'.join(str(n) for n in name)).strip('_')
    return out if out and not out[0].isdigit() else f'c{out}'


def changes_of(update, comps):
    """ The change in each compartment from an event, by name """
    return {name:int(n) for name,n in zip(comps, list(update)) if n != 0}


def numeric_events(model, comps, skip):
    """ The events of an EpiModel: its rates are numbers, or functions of time and state that can't be read """
    out = []
    for kind,rates,updates in [('birth', model.birth_rate_functions, model.birth_event_updates),
                               ('linear', model.linear_rate_functions, model.linear_event_updates),
                               ('quadratic', model.quadratic_rate_functions, model.quadratic_event_updates)]:
        for r,update in zip(rates, updates):
            changes = changes_of(update, comps)
            if callable(r.rate):
                skip.append((f'{kind} event {changes}', 'its rate is a Python function of time; a symbolic model can be read, or a value that changes in time could be an ss.timeseries'))
                continue
            factors = [repr(float(r.rate))] + [comps[getattr(r, c)] for c in ['comp0', 'comp1'] if hasattr(r, c)]
            out.append(dict(kind=kind, rate='*'.join(factors), changes=changes))
    return out


def symbolic_events(model, comps):
    """ The events of a SymbolicEpiModel, with the rates as Python, and the names of its parameters """
    import sympy
    symbols = {c:sympy.Symbol(name) for c,name in zip(model.compartments, comps)}
    found = set()
    for rate in model.linear_rate_functions + model.quadratic_rate_functions + model.birth_rate_functions:
        found |= sympy.sympify(rate).free_symbols
    found -= set(model.compartments) | {model.t}
    params = {p:sympy.Symbol(clean(p)) for p in found}
    def text(rate):
        rate = sympy.sympify(rate).xreplace({**symbols, **params, sympy.pi:sympy.Float(np.pi)})
        return str(rate).replace('Max(', 'max(').replace('Min(', 'min(').replace('Abs(', 'abs(')
    out = []
    for kind,rates,updates in [('birth', model.birth_rate_functions, model.birth_event_updates),
                               ('linear', model.linear_rate_functions, model.linear_event_updates),
                               ('quadratic', model.quadratic_rate_functions, model.quadratic_event_updates)]:
        out += [dict(kind=kind, rate=text(r), changes=changes_of(update, comps)) for r,update in zip(rates, updates)]
    values = {clean(p):model.parameter_values.get(p) for p in found}
    values = {k:(None if v is None else float(v)) for k,v in values.items()}
    return out, values


def stochastic_events(model, comps, skip):
    """ The events of a StochasticEpiModel: a node's transitions, and its links' transmissions at the mean degree """
    events, rates = 0, 1 # How epipack lays out each compartment's entry
    out = []
    k = float(np.mean(model.out_strength))
    for row in model.node_transition_events or []:
        for (_, s, t),r in zip(row[events], row[rates]):
            out.append(dict(kind='linear', rate=f'{float(r)!r}*{comps[s]}', changes={comps[s]:-1, comps[t]:1}))
    for row in model.link_transmission_events or []:
        for (c, s, t),r in zip(row[events], row[rates]):
            out.append(dict(kind='quadratic', rate=f'{float(r)*k!r}*{comps[c]}*{comps[s]}', changes={comps[s]:-1, comps[t]:1}))
    if getattr(model, 'is_network_model', False):
        skip.append(('network', f'its {model.N_nodes} nodes and their links; the model is read as well mixed, at the mean degree, {k:g}'))
    if model.conditional_link_transmission_events:
        skip.append(('conditional link transmissions', 'events that follow another event at once, which are not translated'))
    return out


def model_dict(model, run=None):
    """ The model as a dict that can be written as JSON; run is a dict with the end time and the solver, if any """
    if run is None:
        run = calls.get(id(model))
    comps = [clean(c) for c in model.compartments]
    if len(set(comps)) < len(comps):
        errormsg = f'The compartments {model.compartments} are the same as names in Python ({comps})'
        raise ValueError(errormsg)
    skip = []
    values = {}
    if hasattr(model, 'node_transition_events'):
        events = stochastic_events(model, comps, skip)
        size = model.N_nodes
        dynamic = False
    elif not any(getattr(model, f'{kind}_rate_functions', None) for kind in ['birth', 'linear', 'quadratic']):
        errormsg = f'{type(model).__name__} has no reaction processes to read; models set by their ODEs or matrices are not translated'
        raise TypeError(errormsg)
    elif hasattr(model, 'parameter_values'):
        events, values = symbolic_events(model, comps)
        size = model.initial_population_size
        dynamic = model.correct_for_dynamical_population_size
    else:
        events = numeric_events(model, comps, skip)
        size = model.initial_population_size
        dynamic = model.correct_for_dynamical_population_size
    run = dict(run) if run else None
    y0 = run.pop('y0', None) if run else None
    y0 = y0 if y0 is not None else getattr(model, 'y0', None)
    return dict(
        format     = 'epipack',
        name       = type(model).__name__,
        compartments = comps,
        population = float(size),
        dynamic    = bool(dynamic),
        events     = events,
        parameters = values,
        initial    = None if y0 is None else {c:float(v) for c,v in zip(comps, y0)},
        run        = run,
        incomplete = skip,
    )


def dump(model, filename, run=None):
    """
    Write the model (and the settings of a run, if given or recorded) to a JSON file

    Args:
        model (`EpiModel`): the epipack model
        filename (str): the JSON file to write
        run (dict): the end time as `tmax`, and the `solver`; if None, those recorded
    """
    with open(filename, 'w') as f:
        json.dump(model_dict(model, run=run), f, indent=1)
    return filename


def record():
    """ Record the last call to integrate() or simulate() of each model """
    import epipack
    def recorder(original, solver):
        def run(self, *args, **kwargs):
            arg = args[0] if args else kwargs.get('time_points', kwargs.get('tmax'))
            y0 = getattr(self, 'y0', None)
            calls[id(self)] = dict(tmax=float(np.max(arg)), solver=solver, y0=None if y0 is None else np.array(y0, dtype=float).tolist()) # A simulation leaves y0 at its end
            return original(self, *args, **kwargs)
        return run
    for cls in [epipack.EpiModel, epipack.StochasticEpiModel]:
        for name,solver in dict(integrate='ode', integrate_and_return_by_index='ode', simulate='gillespie').items():
            if hasattr(cls, name):
                setattr(cls, name, recorder(getattr(cls, name), solver))
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
    model = scope[func]() if func else scope['model']
    dump(model, filename)
    print(f'Wrote {filename}: {len(model.compartments)} compartments', file=sys.stderr)
    return


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
