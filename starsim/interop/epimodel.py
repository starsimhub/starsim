"""
Import an EpiModel model

EpiModel (https://www.epimodel.org) is an R package, and its models are R scripts, so
they are read from the JSON written by `dump_epimodel.R`, which runs the script in
EpiModel's own environment with the fitting and simulating functions replaced by ones
that record their arguments. A model of a built-in type (`"SI"`, `"SIS"`, `"SIR"`)
becomes a disease with its transitions; a network fitted by `netest()` becomes an
`ss.PartnerNet` with the same targets (an approximation, since it is not an ERGM); and
modules written in R (`infection.FUN = infect`, ...) are listed, with their source in
`sim.imported.modules`, to be written by hand.
"""
import os
import shutil
import subprocess
import tempfile
import numpy as np
import sciris as sc
import starsim as ss
from .core import Source, build

__all__ = ['load_epimodel']

here = os.path.dirname(os.path.abspath(__file__))
probs = {unit:(lambda v, unit=unit: ss.prob(v, unit)) for unit in ['day', 'week', 'month', 'year']} # A probability per step, by the unit of a step
rates = dict(day=ss.perday, week=ss.perweek, month=ss.permonth, year=ss.peryear)
known = { # The parameters each built-in model reads; any others are listed
    'netsim': {'inf.prob', 'act.rate', 'rec.rate', 'a.rate', 'ds.rate', 'di.rate', 'dr.rate', 'vital'},
    'icm':    {'inf.prob', 'act.rate', 'rec.rate', 'a.rate', 'ds.rate', 'di.rate', 'dr.rate', 'vital'},
    'dcm':    {'inf.prob', 'act.rate', 'rec.rate', 'a.rate', 'ds.rate', 'di.rate', 'dr.rate', 'vital'},
}
modules = dict(netsim='infection.FUN recovery.FUN departures.FUN arrivals.FUN'.split()) # The modules a built-in network model replaces


def dump(filename, rscript=None):
    """ Run `dump_epimodel.R` on an R script or an RDS file, and return the JSON it writes """
    rscript = rscript or shutil.which('Rscript')
    if rscript is None:
        errormsg = f'To read "{filename}", give rscript= (an Rscript with EpiModel installed), or run dump_epimodel.R on it and load the JSON'
        raise FileNotFoundError(errormsg)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'model.json')
        proc = subprocess.run([rscript, os.path.join(here, 'dump_epimodel.R'), os.path.abspath(filename), out], capture_output=True, text=True)
        if proc.returncode or not os.path.exists(out):
            errormsg = f'dump_epimodel.R failed on "{filename}":\n{proc.stderr}'
            raise RuntimeError(errormsg)
        return sc.loadjson(out)


def read(source, rscript=None):
    """ The model as a dict: from a dict, the JSON written by dump_epimodel.R, or an R script or RDS file (which needs R) """
    if isinstance(source, dict):
        return source
    if str(source).lower().endswith('.json'):
        return sc.loadjson(source)
    return dump(source, rscript=rscript)


def mapping(x):
    """ A named list from R, which is [] in JSON when it is empty """
    return x if isinstance(x, dict) else {}


def pyname(name):
    """ An R name as a Python one: inf.prob is inf_prob """
    return name.replace('.', '_')


def value(par):
    """ A parameter as one number, or None if it is a vector (e.g. by time since infection) """
    if sc.isnumber(par) or isinstance(par, bool):
        return par
    if isinstance(par, list) and len(par) == 1:
        return par[0]
    return None


def network(net, src, name):
    """ An ss.PartnerNet with the targets of a network fitted by netest(), and its code """
    n = net['n']
    targets = sc.tolist(net['target_stats'])
    terms = [t for t in net['terms'] if t['term'] != 'offset']
    for t in net['terms']:
        if t['term'] == 'offset':
            src.skip(f'{name}: {t["text"]}', 'a term with a fixed coefficient (e.g. no partnerships within a group) has no declared form; restrict partners with a custom network')
    kw = sc.objdict()
    one = len(terms) == len(targets) # One statistic for each term, in order
    edges = targets[[t['term'] for t in terms].index('edges')] if one and 'edges' in [t['term'] for t in terms] else None
    for i,t in enumerate(terms):
        term, args = t['term'], t['args'] or {}
        if isinstance(args, list): # Positional arguments only
            args = {j:a for j,a in enumerate(args)}
        val = targets[i] if one else None
        if term == 'edges' and val is not None:
            kw.n_contacts = round(2*val/n, 10)
        elif term == 'concurrent' and val is not None and not args:
            kw.concurrency = round(val/n, 10)
        elif term == 'degrange' and val == 0 and 'to' not in args:
            kw.max_contacts = int(args.get('from', args.get(0))) - 1
        elif term == 'nodematch' and val is not None and edges and not args.get('diff') and len(args) == 1:
            attr = str(args.get(0, args.get('attr'))).strip('"')
            values = mapping(net['attributes']).get(attr)
            if values is None:
                src.skip(f'{name}: {t["text"]}', f'the attribute "{attr}" is not set on the network')
                continue
            levels, counts = np.unique(values, return_counts=True)
            src.stratify(**{attr: [str(l) for l in levels]}, split=(counts/counts.sum()).round(10).tolist())
            kw.match = {attr: round(val/edges, 10)}
            src.notes.append(f'{name}: nodematch("{attr}") is the share of partnerships within a level of a stratification "{attr}", split as on the network [guessed: matches the share, not an ERGM fit]')
        else:
            src.skip(f'{name}: {t["text"]}', 'an ERGM term with no counterpart in ss.PartnerNet (it matches the number of partners, concurrency, a maximum, and mixing within a group)')
    if net['dissolution'].replace(' ', '') != '~offset(edges)':
        src.skip(f'{name}: dissolution {net["dissolution"]}', 'partnerships that last longer within a group; ss.PartnerNet has one duration (the first is used)')
    kw.dur = float(sc.tolist(net['duration'])[0])
    if kw.dur.is_integer():
        kw.dur = int(kw.dur)
    code = 'ss.PartnerNet(' + ', '.join(f'{k}={v!r}' for k,v in kw.items()) + ')'
    src.notes.append(f'{name}: {net["formation"]} with targets {targets} and partnerships lasting {kw.dur} steps => {code} [guessed: the same targets, not an ERGM fit]')
    return ss.PartnerNet(**kw, name=name), code


def load_epimodel(source, run=0, unit='day', rscript=None, verbose=None, **kwargs):
    """
    Read an EpiModel model into a sim

    Args:
        source (str/dict): the JSON written by `dump_epimodel.R` (or its contents), or an R script or RDS file, which is dumped with `rscript`
        run (int/str): which call of `netsim()`, `dcm()`, or `icm()` to read, by number or by scenario (`.scenario.id`); the others are kept as scenarios
        unit (str): what one step of the model is ('day', 'week', 'month', or 'year'); EpiModel's rates are per step, so this only names them
        rscript (str): the Rscript to dump an R script with (default: the one on the path)
        verbose (bool): whether to print how each flow was read
        kwargs (dict): passed to `ss.Sim()`

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated

    **Example**:

        # In EpiModel's environment: Rscript dump_epimodel.R intro_sis.R model.json
        sim = ss.load('model.json', format='epimodel')
    """
    d = read(source, rscript=rscript)
    file = source if isinstance(source, str) else d.get('file')
    runs = d['runs']
    ids = [mapping(r['param']).get('.scenario.id', f'run{i}') for i,r in enumerate(runs)]
    r = runs[ids.index(run) if isinstance(run, str) else run]
    kind, typ = r['kind'], r['type']
    param, init, control = mapping(r['param']), mapping(r['init']), mapping(r['control'])
    prob, rate = probs[unit], rates[unit]
    src = Source(name=pyname(typ or 'epimodel').lower(), format='epimodel', file=file)
    src.totals = ['N']
    custom = sc.objdict({k:sc.objdict(m) for k,m in mapping(r['modules']).items() if not m['builtin']})

    # States, as EpiModel's status: s, i, and r (a model of custom modules has whatever its modules give it)
    builtin = typ in ('SI', 'SIS', 'SIR')
    states = [] if not builtin else ['S', 'I'] + (['R'] if typ == 'SIR' else [])
    i_num = value(init.get('i.num', 0)) or 0
    r_num = value(init.get('r.num', 0)) or 0
    if kind == 'netsim':
        n = r['networks'][0]['n'] if r['networks'] else None
        s_num = None if n is None else n - i_num - r_num
    else:
        s_num = value(init.get('s.num', 0)) or 0
    for state,num in zip(states, [s_num, i_num, r_num]):
        src.add_state(state, init=num)
    for k in (init if builtin else []):
        if k not in ['i.num', 'r.num', 's.num'] and not k.endswith('.g2'):
            src.skip(f'init {k}', 'initial conditions other than numbers in each status (e.g. a status for each node) are by hand')

    # Parameters and flows of the built-in models
    if builtin:
        if any(k.endswith('.g2') for k in param) or any(k.endswith('.g2') for k in init):
            src.skip('two groups', 'the parameters ending in .g2 are a second group: stratify(group=[1, 2]) with the rates by group, by hand; group 1 is imported')
        for k in param:
            if k not in known[kind] and not k.endswith('.g2') and not k.startswith('.'):
                src.skip(f'param {k}', 'a parameter the built-in model does not read (e.g. inter.eff for an intervention, or one for a custom module)')
        vals = {k:value(v) for k,v in param.items()}
        for k,v in vals.items():
            if k in known[kind] and v is None:
                src.skip(f'param {k}', 'a parameter that varies (by time since infection, or several values for a sensitivity analysis): its first value is used')
                vals[k] = sc.tolist(param[k])[0]
        inf_prob, act_rate = vals.get('inf.prob'), vals.get('act.rate', 1)
        if kind == 'netsim': # The chance per partnership per step, over act.rate acts
            src.add_par('beta', prob(round(1 - (1 - inf_prob)**act_rate, 12)), doc=f'1 - (1 - inf.prob)^act.rate per partnership per step, with inf.prob = {inf_prob} and act.rate = {act_rate}')
            src.add_flow('S', 'I', 'beta * S * I / N', per='total', name='infection')
        else: # A rate of acts with random people, each with a chance of infection
            src.add_par('inf_prob', inf_prob, doc='chance of infection per act')
            src.add_par('act_rate', rate(act_rate), doc='acts per person per step')
            src.add_flow('S', 'I', 'inf_prob * act_rate * S * I / N', per='total', name='infection')
            if kind == 'icm':
                src.notes.append('icm(): acts between random pairs, written as a rate of infection inf_prob * act_rate * I / N [guessed: the same mean, not the same draws]')
        step = rate if kind == 'dcm' else prob # dcm()'s rates are rates; the stochastic models' are chances per step
        if typ != 'SI' and vals.get('rec.rate'):
            src.add_par('rec_rate', step(vals['rec.rate']), doc='recovery per step')
            src.add_flow('I', 'S' if typ == 'SIS' else 'R', 'rec_rate', name='recovery')
        if vals.get('vital') or any(vals.get(k) for k in ['a.rate', 'ds.rate', 'di.rate', 'dr.rate']):
            if vals.get('a.rate'):
                src.add_par('a_rate', rate(vals['a.rate']), doc='arrivals per person per step')
                src.add_flow(None, 'S', 'a_rate * N', per='total', name='arrival')
            for state in states:
                k = f'd{state.lower()}.rate'
                if vals.get(k):
                    src.add_par(pyname(k), step(vals[k]), doc=f'departures from {state} per step')
                    src.add_flow(state, None, pyname(k), name=f'departure_{state}')
        for k in modules.get(kind, []):
            if k in custom:
                src.skip(k, f'replaced by {custom[k].name}(), which is written in R; its rates are imported as the built-in ones')
    else:
        src.skip(f'type {typ}', 'not a built-in model: every module is written in R, so only the networks are imported (the parameters are in sim.imported.pars, and the scenarios in sim.imported.scenarios)')
        src.notes.append(f'init {init} (for the custom modules)')
        for k,v in param.items():
            if value(v) is not None and not k.startswith('.') and not isinstance(v, str):
                src.notes.append(f'param {k} = {value(v)} (for the custom modules)')

    # Custom modules, written in R
    for k,m in custom.items():
        if not (k in modules.get(kind, []) and builtin):
            src.skip(k, f'{m.name}(), a module written in R: write it as transitions, products, and deliveries (its source is in sim.imported.modules)')

    # Networks
    cfg = src.config
    if kind == 'netsim':
        nets = sc.objdict()
        codes = []
        for i,net in enumerate(r['networks']):
            name = 'net' if len(r['networks']) == 1 else f'net{i+1}'
            nets[name], code = network(net, src, name)
            codes.append(f'{name}={code}')
            if net['d_rate']:
                src.notes.append(f'{name}: d.rate = {net["d_rate"]} needs nothing: ss.PartnerNet allows for partnerships ended by departures')
        if len(nets) == 1:
            cfg.networks = nets[0]
            src.code['networks'] = codes[0].split('=', 1)[1]
        else:
            cfg.networks = list(nets.values())
            src.code['networks'] = 'dict(' + ', '.join(codes) + ')'
            src.notes.append(f'{len(nets)} networks: transmission over each, with the same beta [guessed]')
        cfg.n_agents = r['networks'][0]['n']
        cfg.solver = 'abm'
    elif kind == 'icm':
        cfg.solver = 'binomial'
    else:
        cfg.solver = 'ode'
        if control.get('dt') not in (None, 1):
            cfg.dt = getattr(ss, f'{unit}s')(control['dt']) # e.g. ss.days(0.5)

    # Time: EpiModel's time 1 is the start and nsteps the end, so it runs for nsteps - 1 units
    nsteps = control.get('nsteps', 100)
    cfg.unit = unit
    cfg.start = 0
    cfg.stop = int(nsteps) - 1
    nsims = control.get('nsims', 1)
    if nsims and nsims > 1:
        src.notes.append(f'EpiModel ran {nsims} simulations: sim.run(n_runs={nsims})')

    sim = build(src, verbose=verbose, **kwargs)
    for dis in sim.pars.diseases: # EpiModel's infection is exposure to I by definition, not a reading of a rate
        tr = dis.transitions.get('infection')
        if tr is not None:
            tr.guessed.pop('import', None)
    sim.imported.modules = sc.objdict({m.name:m.source for m in custom.values()})
    sim.imported.functions = sc.objdict(d.get('functions') or {})
    sim.imported.pars = sc.objdict({k:v for k,v in param.items() if not k.startswith('.')})
    sim.imported.scenarios = sc.objdict()
    for id,other in zip(ids, runs):
        changed = {k:v for k,v in mapping(other['param']).items() if not k.startswith('.') and param.get(k) != v}
        sim.imported.scenarios[id] = changed
    return sim
