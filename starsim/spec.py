"""
Read a model spec (written by `sim.to_spec()`) back into a sim
"""
import ast
import sciris as sc
import starsim as ss

__all__ = ['from_spec']

# Typed quantities that can be written inside a rate, e.g. perday(0.5)
time_names = ['days', 'weeks', 'months', 'years', 'perday', 'perweek', 'permonth', 'peryear',
              'probperday', 'probperweek', 'probpermonth', 'probperyear', 'freqperday', 'freqperweek', 'freqpermonth', 'freqperyear']

# Functions that can be written inside a rate, by their name in the spec
func_names = ['exp', 'log', 'sqrt', 'absolute', 'sin', 'cos', 'minimum', 'maximum']
comparisons = {ast.Lt:'<', ast.LtE:'<=', ast.Gt:'>', ast.GtE:'>='}


def quantity(val, kind=None):
    """ A value from the model spec: a typed quantity, a distribution, a value by stratum, or a plain number """
    if isinstance(val, dict):
        if 'type' in val:
            unit = val['unit']
            if unit is None:
                return val['value']
            if val['type'] == 'dur':
                return getattr(ss, f'{unit}s')(val['value'])
            prefix = '' if val['type'] == 'per' else val['type']
            return getattr(ss, f'{prefix}per{unit}')(val['value'])
        elif 'dist' in val:
            pars = {k:quantity(v) for k,v in val.items() if k != 'dist'}
            return getattr(ss, val['dist'])(**pars)
        elif 'timeseries' in val:
            times = list(val['timeseries'].keys())
            values = [quantity(v) for v in val['timeseries'].values()]
            kw = dict(period=quantity(val['period'])) if 'period' in val else {}
            return ss.timeseries(dict(zip(times, values)), mode=val['mode'], **kw)
        elif 'by' in val:
            dims = val['by']
            if 'array' in val:
                return ss.by(val['array'], dims=dims)
            elif 'table' in val:
                return ss.by({col:quantity(v) for col,v in val['table'].items()})
            elif 'values' in val:
                return ss.by(**{dims[0]:quantity(val['values'])})
            return ss.by(**{dim:quantity(v) for dim,v in val['product'].items()})
        elif 'seasonal' in val:
            fields = val['seasonal']
            return ss.seasonal(mean=quantity(fields['mean']), amplitude=fields['amplitude'], peak=fields['peak'])
        unknown(f'the value {val}')
    elif isinstance(val, list):
        return [quantity(v) for v in val]
    return val


def unknown(what):
    errormsg = f'ss.load() cannot yet read {what} from a model spec'
    raise NotImplementedError(errormsg)


class Parser:
    """ Read a rate or a result written as text, e.g. "beta * W.n / (W.n + K)", allowing only the names the model declares """
    def __init__(self, names):
        self.names = names
        return

    def __call__(self, text):
        if text is None:
            return None
        return self.eval(ast.parse(text, mode='eval').body)

    def eval(self, node):
        ev = self.eval
        if isinstance(node, ast.Constant) and sc.isnumber(node.value):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id not in self.names:
                errormsg = f'The model spec uses "{node.id}", which is not a state, parameter, or transition of the module'
                raise ValueError(errormsg)
            return self.names[node.id]
        elif isinstance(node, ast.Attribute) and node.attr in ['n', 'frac']:
            return getattr(ev(node.value), node.attr)
        elif isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            ops = {ast.Add:lambda: a + b, ast.Sub:lambda: a - b, ast.Mult:lambda: a * b, ast.Div:lambda: a / b, ast.Pow:lambda: a ** b}
            if type(node.op) in ops:
                return ops[type(node.op)]()
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in comparisons:
            a, b = ss.transitions.as_expr(ev(node.left)), ev(node.comparators[0])
            return {'<':a.__lt__, '<=':a.__le__, '>':a.__gt__, '>=':a.__ge__}[comparisons[type(node.ops[0])]](b)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            args = [ev(arg) for arg in node.args]
            if name in time_names:
                return getattr(ss, name)(*args)
            elif name in func_names:
                return ss.transitions.Func(name, *args)
        errormsg = f'ss.load() cannot read "{ast.unparse(node)}" in a model spec'
        raise ValueError(errormsg)


def disease(spec):
    """ One declared disease from its spec """
    if spec['type'] != 'Disease':
        unknown(f'a module of type {spec["type"]}')
    if spec['functions']:
        unknown(f'functions ({sc.strjoin(f["name"] for f in spec["functions"])})')

    # States
    states = sc.objdict()
    for s in spec['states']:
        init = s['init'] if isinstance(s['init'], str) else quantity(s['init']) # An expression of the parameters is read once they are
        people = None if f'states.{s["name"]}.people' in spec['guessed'] else s['people'] # Worked out again, as it was the first time
        states[s['name']] = ss.State(name=s['name'], init=init, people=people, infectious=s['infectious'], rel_sus=s['rel_sus'], rel_trans=s['rel_trans'], label=s['label'])

    # Parameters
    pars = sc.objdict()
    for p in spec['pars']:
        if p['prior'] is not None:
            unknown(f'the prior of parameter {p["name"]}')
        pars[p['name']] = ss.par(quantity(p['value']), bounds=p['bounds'], source=p['source'], doc=p['doc'], name=p['name'])
    parse = Parser({**states, **pars})
    for state in states.values():
        if isinstance(state.init, str):
            state.init = parse(state.init)

    # Transitions
    def find_par(ref):
        assert ref[0] == 'par', ref
        return pars[ref[1]]
    kwargs = sc.objdict()
    names = sc.objdict()
    unnamed = []
    for t in spec['transitions']:
        src = [states[n] for n in t['src']] if t['src'] else None
        dst = [states[n] for n in t['dst']] if t['dst'] else None
        tr = dict(rate=parse(t['rate']), p=None if t['p'] is None else [parse(p) for p in t['p']], stages=t['stages'], label=t['label'])
        dur = t['dur']
        if dur is not None:
            if dur['dist'] == 'expon' and 'mean' in dur and isinstance(dur['mean'], list):
                tr['dur'] = find_par(dur['mean'])
            elif dur['dist'] == 'expon' and 'rate' in dur:
                tr['rate'] = find_par(dur['rate'])
            else: # A distribution, held by the parameter that has it as its value
                held = [p['name'] for p in spec['pars'] if p['value'] == dur]
                tr['dur'] = pars[held[0]] if held else quantity(dur)
        trans = ss.Transition(ss.null if src is None else src, ss.null if dst is None else dst, **tr)
        trans.migration = t.get('migration', False)
        if f'transitions.{t["name"]}.name' in spec['guessed']: # Named from its endpoints again
            unnamed.append(trans)
        else:
            kwargs[t['name']] = trans
        names[t['name']] = trans

    # Results and outcomes, which may refer to the transitions
    parse = Parser({**states, **pars, **names})
    for r in spec['results']:
        if r['type'] == 'result':
            kwargs[r['name']] = parse(r['source'])
        else:
            if r['likelihood'] is not None:
                unknown(f'the likelihood of outcome {r["name"]}')
            kwargs[r['name']] = ss.outcome(parse(r['source']), delay=quantity(r['delay']), duration=quantity(r['duration']), every=quantity(r['every']), name=r['name'])

    dis = ss.Disease(*states.values(), *unnamed, **kwargs, init_prev=spec['init_prev'], name=spec['name'])
    graph = dis.graph
    for key in spec['guessed']: # What was guessed in the model that was saved, and is declared here (the states, a transition's source)
        parts = key.split('.')
        if parts[0] == 'transitions' and parts[1] in graph.transitions:
            graph.transitions[parts[1]].guessed.setdefault(parts[2], 'in the model spec')
        elif parts[0] != 'transitions':
            graph.guessed.setdefault(key, 'in the model spec')
    for strat in spec['stratifications']:
        dis.stratify(**strat_kwargs(strat))
    return dis


def strat_kwargs(strat):
    """ The arguments to stratify() for one stratification in the spec """
    if strat['mobility'] is not None:
        unknown(f'the movement between places of "{strat["name"]}"')
    if isinstance(strat['split'], str):
        unknown(f'the joint split of "{strat["name"]}"')
    out = {strat['name']:strat['levels'], 'split':strat['split'], 'adjust':strat['adjust'] or None, 'kind':strat['kind']}
    if strat['states'] is not None:
        out['states'] = strat['states']
    return out


def from_spec(spec):
    """
    Build a sim from a model spec, as written by `sim.to_spec()`

    Diseases declared from states and transitions, their stratifications, and the run
    settings can be read back; networks, interventions, modifiers, functions, and
    movement between places cannot yet.

    Args:
        spec (dict/str): the spec, or the name of a JSON file holding one

    Examples:
        ```python
        sim.to_spec('sim.json')
        sim2 = ss.load('sim.json')
        assert sim2.hash() == sim.hash()
        ```
    """
    if not isinstance(spec, dict):
        spec = sc.loadjson(spec)
    if not str(spec.get('schema', '')).startswith('lingua-franca/'):
        errormsg = 'This is not a model spec: it has no "schema" of "lingua-franca/..."'
        raise ValueError(errormsg)
    for key in ['networks', 'modifiers', 'interventions']:
        if spec.get(key):
            unknown(key)
    modules = [disease(mod) for mod in spec['modules']]
    cfg = spec['config']
    start, stop = cfg['start'], cfg['stop']
    if cfg['unit'] is not None and sc.isnumber(start) and sc.isnumber(stop) and not start: # No dates: durations in the sim's unit
        start = None
    kwargs = dict(solver=cfg['solver'], integrator=cfg['integrator'], approx=cfg['approx'], unit=cfg['unit'], start=start, stop=stop,
                  dt=cfg['dt'], total_pop=cfg['total_pop'], n_agents=cfg['n_agents'], rand_seed=cfg['rand_seed'])
    sim = ss.Sim(modules, **{k:v for k,v in kwargs.items() if v is not None})
    for strat in spec['stratify']:
        sim.stratify(**strat_kwargs(strat))
    return sim
