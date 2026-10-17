"""
Import a summer2 model

summer2 (https://github.com/monash-emu/summer2) builds a model in Python, so it is read
either as a live `CompartmentalModel` (if summer2 is installed alongside Starsim) or as
the JSON written by `dump_summer2.py`, which runs in summer2's own environment. Each
flow becomes one transition with the matching kind of rate, each `stratify_with()` a
stratification (with its population split, flow adjustments, infectiousness
adjustments, and mixing matrix), and the derived outputs can be worked out from the
results with `ss.interop.summer2.outputs()`.
"""
import numpy as np
import sciris as sc
import starsim as ss
from .core import Source, build

__all__ = ['load_summer2', 'outputs']

skipped_kinds = dict(ReplacementBirthFlow='births that replace the deaths in each step, which have no rate of their own')


def read(source):
    """ The model as a dict: from a dict, a JSON file, or a live summer2 model """
    if isinstance(source, dict):
        return source
    elif hasattr(source, 'compartments') and hasattr(source, 'flows') and hasattr(source, '_stratifications'):
        from . import dump_summer2 # Needs only summer2, which is installed if a model was made
        return dump_summer2.model_dict(source)
    return sc.loadjson(source)


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


def load_summer2(source, verbose=None):
    """
    Read a summer2 model into a sim

    Args:
        source (str/dict/CompartmentalModel): the JSON file written by `dump_summer2.py` (or its contents), or the model itself
        verbose (bool): whether to print how each flow was read

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated

    **Example**:

        # In the summer2 environment: python dump_summer2.py my_model.py:build_model model.json
        sim = ss.load('model.json', format='summer2')
    """
    d = read(source)
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
            pname, param = param['par'], param['value']
        if not sc.isnumber(param):
            src.skip(f'flow {name}', f'its parameter is a function ({param.get("function") if isinstance(param, dict) else param}); a value that varies in time could be an ss.timeseries')
            continue
        base[name] = param
        src.add_par(pname, param, doc=f'summer2 {kind} "{name}"')
        sources = f['sources']
        sick = ' + '.join(infectious)
        if kind == 'InfectionFrequencyFlow':
            flows = [(sources[0], f['dest'], f'{pname} * ({sick}) / N', 'person')]
        elif kind == 'InfectionDensityFlow':
            flows = [(sources[0], f['dest'], f'{pname} * ({sick})', 'person')]
        elif kind in ['TransitionFlow', 'DeathFlow']:
            flows = [(s, f['dest'], pname, 'person') for s in sources]
        elif kind == 'AbsoluteFlow':
            flows = [(sources[0], f['dest'], pname, 'total')]
        elif kind == 'CrudeBirthFlow':
            flows = [(None, f['dest'], f'{pname} * N', 'total')]
        elif kind == 'ImportFlow':
            flows = [(None, f['dest'], pname, 'total')]
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
        src.config.networks = ss.contacts(**mixing)
        src.code['networks'] = 'ss.contacts(' + ', '.join(f'{k}={np.asarray(v).tolist()!r}' for k,v in mixing.items()) + ')'
    outs = d.get('outputs') or {}
    for c,val in byinit.items():
        src.states[c].init = val
    sim = build(src, verbose=verbose)
    sim.imported.outputs = outs
    if outs:
        sim.imported.choices.append(f'  {len(outs)} derived outputs: worked out from the results by ss.interop.summer2.outputs(sim)')
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
    the other outputs.

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
            val = sum(select(res[f'n_{c}'], req.get('strata') or {}) for c in req['compartments'])
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
