"""
Import an epymorph model

epymorph (https://github.com/NAU-CCL/Epymorph) binds a disease model, movement, geography,
parameters, and initial numbers into a RUME, and needs its own environment, so a RUME is
read from the JSON written by this file run as a script (see below). Each compartment
becomes a state and each edge a flow, with `Max(1, S + I + R)` (epymorph's guard against
dividing by zero) read as the total; each node becomes a level of a stratification by
place. Rates are those of the people present in a node, as in a stratification by place.

Parameters are read as epymorph resolves them, whether they were given as numbers or came
from an ADRIO, as a value, a value by place, or an `ss.timeseries`; one that varies both
by place and in time is taken at each place's mean. Forks become one flow for each edge.
A RUME with several strata is refused, since exposure within a stratum, which divides by
that stratum's people, does not move between places here.

Movement in epymorph is code, so its clauses are run, not read: over the first four
weeks, the people each clause moves from each node to each other node are averaged, as a
share of the node's people, and multiplied by the time they are away. This gives commuting
(`ss.geo.Mobility(kind='commute')`), the share of their time that people spend in each
other place, with the time of day and the day of the week averaged out. Two things are
listed as not translated: a clause that never returns people, and compartments tagged
`immobile`, which commute here like the others.

epymorph draws each event as Poisson in each of its tau steps, capped at the number in
the compartment, so the sim is `solver='tau'`, which leaps in the same way. A chain
binomial gives peaks far too high at epymorph's steps of a third of a day or more.

Writing the JSON: run as a script, this file runs where epymorph is installed, and needs
neither Starsim nor Sciris, so it can be copied into any epymorph environment. Run it on a
function that builds the RUME, or on a script that makes one called `rume`:

    python epymorph.py my_model.py:build_rume model.json
    python epymorph.py my_model.py model.json

or call `dump(rume, 'model.json')` from Python.

Used with: compartments; declared modules
"""
import os
import sys
if __name__ == '__main__': # Run as a script, this folder would hide epymorph itself (with this file, epymorph.py)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path = [p for p in sys.path if os.path.abspath(p or '.') != here]
import json
import runpy
import numpy as np
if __package__: # As part of Starsim; run as a script in epymorph's environment, the rest needs neither Sciris nor Starsim
    import sciris as sc
    import starsim as ss
    from .core import Source, build

__all__ = ['load_epymorph', 'dump', 'rume_dict']

sampled_days = 28 # The days over which movement is averaged: whole weeks, for clauses that differ by the day


def load_epymorph(source, solver='tau', verbose=None, **kwargs):
    """
    Read an epymorph RUME into a sim

    Args:
        source (str/dict): the JSON file written by `epymorph.py` (or its contents)
        solver (str): the solver
        verbose (bool): whether to print how each edge was read
        kwargs (dict): passed to `ss.Sim()`

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated

    **Example**:

        # In the epymorph environment: python epymorph.py my_model.py model.json
        sim = ss.import_from('model.json', format='epymorph')
    """
    d = source if isinstance(source, dict) else sc.loadjson(source)
    src = Source(name='epymorph', format='epymorph', file=source if isinstance(source, str) else None)
    src.incomplete = [tuple(item) for item in d.get('incomplete', [])]
    src.totals = ['N']
    comps = d['compartments']
    places = d['places']
    n = len(places)
    init = np.array(d['initial'], dtype=float) # Places by compartments
    totals = init.sum(axis=1)
    dates = np.arange(np.datetime64(d['start'], 'D'), np.datetime64(d['start'], 'D') + np.timedelta64(d['days'], 'D'))

    def value(name, p):
        """ A parameter's value in Starsim """
        kind, val = p['kind'], p['value']
        if kind == 'number':
            return val
        elif kind == 'place':
            return ss.by(val, dims=['patch'])
        elif kind == 'time_place':
            src.skip(f'parameter {name}', 'it varies both in time and by place; it is taken at the mean for each place')
            return ss.by(np.mean(val, axis=0).tolist(), dims=['patch'])
        vals = np.array(val, dtype=float)
        changes = np.flatnonzero(vals[1:] != vals[:-1]) + 1
        return ss.timeseries({str(dates[i]):float(vals[i]) for i in [0, *changes]})

    # States: the rest of each place is in the state that holds most of it, as Starsim fills the first state
    by_state = dict(zip(comps, init.T))
    rest = max(comps, key=lambda c: by_state[c].sum())
    for c in [rest] + [c for c in comps if c != rest]:
        vals = by_state[c]
        src.add_state(c, init=None if c == rest or not vals.sum() else ss.by(vals.tolist(), dims=['patch']) if n > 1 else float(vals.sum()))

    # Flows
    for e in d['edges']:
        s, t = e['source'], e['target']
        name = f'{s}_to_{t}' if s and t else f'{s}_out' if s else f'{t}_in'
        name = name if name not in [f.name for f in src.flows] else f'{name}_{len(src.flows)}'
        src.add_flow(s, t, e['rate'], per='total', name=name)

    for name,p in d['parameters'].items():
        src.add_par(name, value(name, p), doc=f'epymorph parameter "{name}"')

    # Places, and the movement between them
    if n > 1:
        away = np.array(d['movement'], dtype=float) # Without movement this is all zeros, which keeps the places apart, as epymorph's nodes are; with no mobility= they would mix
        src.stratify(patch=places, split=(totals/totals.sum()).tolist(), mobility=ss.geo.Mobility(matrix=away.tolist(), kind='commute'))
        moved = away.sum(axis=1)
        src.notes.append(f'places        the {n} nodes as the places "{places[0]}" to "{places[-1]}", ' + (f'with the movement clauses run and averaged into commuting: people spend {moved.min():.3g} to {moved.max():.3g} of their time away' if away.any() else 'with no movement'))

    # Run settings
    src.config.update(unit='day', dt=1, start=d['start'], stop=str(dates[-1] + np.timedelta64(1, 'D')), solver=solver, total_pop=float(totals.sum()), use_aging=False)
    return build(src, verbose=verbose, **kwargs)


#%% Writing the RUME to JSON, in epymorph's own environment (needs neither Sciris nor Starsim)

def edge_rate(rate, compartments):
    """ A rate as Python, with epymorph's Max(1, S + I + R) read as the total, N """
    import sympy
    everyone = sum(compartments)
    def is_total(e):
        return isinstance(e, sympy.Max) and len(e.args) == 2 and 1 in e.args and everyone in e.args
    rate = rate.replace(is_total, lambda e: sympy.Symbol('N'))
    return str(rate.xreplace({sympy.pi:sympy.Float(np.pi)})).replace('Max(', 'max(').replace('Min(', 'min(').replace('Abs(', 'abs(')


def parameter(vals):
    """ A parameter (days by places) as JSON, in the simplest form that holds it """
    if np.ptp(vals) == 0:
        return dict(kind='number', value=float(vals[0, 0]))
    elif np.ptp(vals, axis=0).max() == 0:
        return dict(kind='place', value=vals[0].tolist())
    elif np.ptp(vals, axis=1).max() == 0:
        return dict(kind='time', value=vals[:, 0].tolist())
    return dict(kind='time_place', value=vals.tolist())


def movement(rume, data, rng, people, skip):
    """ The share of their time that the people of each place (rows) spend in each other place: the clauses run for the first weeks, and averaged """
    from epymorph.attribute import ModuleNamespace
    from epymorph.simulation import NEVER, Context, resolve_tick_delta, simulation_clock
    from epymorph.strata import gpm_strata
    days = min(sampled_days, rume.time_frame.days)
    ticks = [tick for tick in simulation_clock(rume.time_frame, rume.tau_step_lengths) if tick.day < days]
    steps = np.tile(rume.tau_step_lengths, rume.time_frame.days + days) # Long enough for trips that outlast the days sampled
    away = np.zeros((rume.scope.nodes, rume.scope.nodes))
    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.nan_to_num(1/people, posinf=0.0)
    for strata,model in rume.mms.items():
        namespace = ModuleNamespace(gpm_strata(strata), 'mm')
        for clause in model.clauses:
            ctx = Context.of(namespace.to_absolute(clause.clause_name), data, rume.scope, rume.time_frame, rume.ipm, rng)
            c = clause.with_context_internal(ctx)
            if c.returns == NEVER:
                skip.append((f'movement clause {clause.clause_name}', 'it moves people for good, which is not translated'))
                continue
            for tick in ticks:
                if c.is_active(tick):
                    back = resolve_tick_delta(rume.num_tau_steps, tick, c.returns)
                    movers = np.array(c.evaluate(tick), dtype=float)
                    np.fill_diagonal(movers, 0)
                    away += (movers.T*share).T * steps[tick.sim_index:back].sum()
    return away/days


def rume_dict(rume, seed=0):
    """
    The RUME as a dict that can be written as JSON

    Args:
        rume (`RUME`): the epymorph model
        seed (int): the seed for what is drawn in resolving it (initial locations, and the movers from each clause)
    """
    from epymorph.compartment_model import BIRTH, DEATH
    if len(rume.strata) > 1:
        errormsg = f'A RUME with several strata is not translated: each stratum divides by its own people, and exposure to those of a stratum does not move between places here, so the {len(rume.strata)} strata would not spread as they do in epymorph'
        raise TypeError(errormsg)
    rng = np.random.default_rng(seed)
    data = rume.evaluate_params(rng)
    initial = np.array(rume.initialize(data, rng), dtype=float)
    ipm = rume.ipm
    skip = []
    days, nodes, tau = rume.time_frame.days, rume.scope.nodes, rume.num_tau_steps
    compartments = list(ipm.symbols.all_compartments)
    names = [str(c) for c in compartments]

    edges = []
    for e in ipm.events:
        source, target = e.compartment_from, e.compartment_to
        rate = edge_rate(e.rate, compartments)
        edges.append(dict(source=None if source == BIRTH else str(source), target=None if target == DEATH else str(target), rate=rate))
        if 'max(' in rate:
            skip.append((f'edge {source} -> {target}', f'its rate, {rate}, divides by some of the people, which is kept as counts in each place, and does not carry exposure between places; in movement it should be checked'))

    # The requirements, as they are at each node at the first tau step of each day
    series = data.resolve_txn_series(ipm.requirements_dict.items(), tau)
    reqs = [str(r) for r in ipm.symbols.all_requirements]
    vals = np.array([[float(v) for v in next(series)] for _ in range(days*tau*nodes)]).reshape(days, tau, nodes, len(reqs))[:, 0]
    parameters = {name:parameter(vals[:, :, i]) for i,name in enumerate(reqs)}

    immobile = [c.name.full for c in ipm.compartments if 'immobile' in c.tags]
    if immobile:
        skip.append(('immobile compartments', f'{immobile} are immobile in epymorph; here everyone commutes alike'))
    labels = rume.scope.labels if hasattr(rume.scope, 'labels') else rume.scope.node_ids
    return dict(
        format       = 'epymorph',
        name         = type(ipm).__name__,
        compartments = names,
        places       = [str(p) for p in labels],
        edges        = edges,
        parameters   = parameters,
        initial      = initial.tolist(),
        start        = str(rume.time_frame.start_date),
        days         = int(days),
        movement     = movement(rume, data, rng, initial.sum(axis=1), skip).tolist(),
        incomplete   = skip,
    )


def dump(rume, filename, seed=0):
    """
    Write the RUME to a JSON file

    Args:
        rume (`RUME`): the epymorph model
        filename (str): the JSON file to write
        seed (int): the seed for what is drawn in resolving it
    """
    with open(filename, 'w') as f:
        json.dump(rume_dict(rume, seed=seed), f, indent=1)
    return filename


def main(target, filename):
    """ Build the RUME from a function ("file.py:function") or a script (with a variable "rume"), and write it """
    path, _, func = target.partition(':')
    scope = runpy.run_path(path, run_name='__main__' if not func else '__dump__')
    rume = scope[func]() if func else scope['rume']
    dump(rume, filename)
    print(f'Wrote {filename}: {rume.ipm.num_compartments} compartments, {rume.ipm.num_events} edges, {rume.scope.nodes} places', file=sys.stderr)
    return


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
