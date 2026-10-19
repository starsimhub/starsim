"""
The model spec (written by `sim.to_spec()`, for `sim.hash()` and `ss.diff()`) as code (`ss.to_code()`), and the parser of expressions in it
"""
import ast
import textwrap
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['to_code', 'diff']

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
        elif isinstance(node, ast.Attribute) and node.attr in ['n', 'frac', 'total']:
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


def named_args(ref, keys):
    """ The arguments that looked up named data, from where the spec says it came from """
    return {k:ref[k] for k in keys if ref.get(k) is not None}


geo_keys = ['level', 'year', 'places']
contacts_keys = ['source', 'layer']
people_keys = ['year', 'revision']


def strat_kwargs(strat):
    """ The arguments to stratify() for one stratification in the spec """
    if strat['mobility'] is not None:
        unknown(f'the movement between places of "{strat["name"]}"')
    split = strat['split']
    if isinstance(split, str): # A split over several dimensions, written as by([[...]], dims=[...])
        split = joint_split(split)
    levels = strat['levels']
    if strat.get('geo'): # A named geography, looked up again; the split is as the spec has it
        levels = ss.geo(strat['geo']['place'], **named_args(strat['geo'], geo_keys))
    out = {strat['name']:levels, 'split':split, 'adjust':strat['adjust'] or None, 'kind':strat['kind']}
    if strat.get('split_from'): # Taken from the named population again
        out.pop('split')
    if strat['states'] is not None:
        out['states'] = strat['states']
    if strat.get('where') is not None:
        out['where'] = strat['where']
    return out


def joint_split(text):
    """ A joint split from the model spec, read as literals (so no code is run) """
    node = ast.parse(text, mode='eval').body
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'by'):
        unknown(f'the split "{text}"')
    args = [ast.literal_eval(a) for a in node.args]
    kwargs = {k.arg:ast.literal_eval(k.value) for k in node.keywords}
    return ss.by(*args, **kwargs)


class Code(str):
    """ Python to write as it is, e.g. an object an importer passed to the sim """
    pass


def code_value(val):
    """ The Python for a value in the model spec: the inverse of quantity() """
    if isinstance(val, Code):
        return str(val)
    elif isinstance(val, ss.by) and val.array is not None:
        return f'ss.by({np.asarray(val.array).tolist()!r}, dims={list(val.dims)!r})'
    elif isinstance(val, dict):
        if 'type' in val:
            unit = val['unit']
            if unit is None:
                return repr(val['value'])
            if val['type'] == 'dur':
                return f'ss.{unit}s({val["value"]!r})'
            prefix = '' if val['type'] == 'per' else val['type']
            return f'ss.{prefix}per{unit}({val["value"]!r})'
        elif 'dist' in val:
            args = ', '.join(f'{k}={code_value(v)}' for k,v in val.items() if k != 'dist')
            return f'ss.{val["dist"]}({args})'
        elif 'timeseries' in val:
            items = ', '.join(f'{time_code(k)}: {code_value(v)}' for k,v in val['timeseries'].items())
            period = f', period={code_value(val["period"])}' if 'period' in val else ''
            return f"ss.timeseries({{{items}}}, mode={val['mode']!r}{period})"
        elif 'by' in val:
            dims = val['by']
            if 'array' in val:
                return f'ss.by({val["array"]!r}, dims={dims!r})'
            elif 'table' in val:
                return 'ss.by({' + ', '.join(f'{k!r}: {code_value(v)}' for k,v in val['table'].items()) + '})'
            elif 'values' in val:
                return f'ss.by({dims[0]}={code_value(val["values"])})'
            return 'ss.by(' + ', '.join(f'{k}={code_value(v)}' for k,v in val['product'].items()) + ')'
        elif 'seasonal' in val:
            f = val['seasonal']
            return f'ss.seasonal(mean={code_value(f["mean"])}, amplitude={f["amplitude"]!r}, peak={f["peak"]!r})'
        return '{' + ', '.join(f'{k!r}: {code_value(v)}' for k,v in val.items()) + '}' # A plain dict, e.g. adjust=
    elif isinstance(val, list):
        return '[' + ', '.join(code_value(v) for v in val) + ']'
    return repr(val)


def time_code(text):
    """ The Python for a time written as text in the model spec (a key of a time series, or the start of a modifier): a number, a duration (e.g. "days(5)", or "day" for one), or a date """
    try:
        return repr(ast.literal_eval(text))
    except (ValueError, SyntaxError):
        pass
    unit, _, value = text.partition('(')
    if unit.removesuffix('s') in ['day', 'week', 'month', 'year']:
        return f"ss.{unit.removesuffix('s')}s({value.removesuffix(')') or 1})"
    return repr(text)


def code_expr(text):
    """ The Python for a rate or result in the model spec: functions from NumPy, and typed quantities from Starsim """
    if text is None:
        return None
    node = ast.parse(text, mode='eval').body
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id in func_names:
                n.func = ast.Attribute(ast.Name('np'), n.func.id)
            elif n.func.id in time_names or hasattr(ss, n.func.id): # Every Starsim name, e.g. ss.by() or ss.seasonal()
                n.func = ast.Attribute(ast.Name('ss'), n.func.id)
    return ast.unparse(node)


def func_code(fn, funcs, left):
    """ A function in the model spec as the name of its definition, which is added to funcs (with the values it closes over); or None, listed in left, if its source is not a plain definition """
    source = fn.get('source')
    text = textwrap.dedent(source).strip() if source else ''
    if not text.startswith('def '):
        left.append(f'the function {fn["fn"]} (written as None): define it and pass it in place of None')
        return None
    values = [f'{k} = {v!r}' for k,v in (fn.get('values') or {}).items() if not k.startswith('default_')] # Defaults are in the source
    block = '\n'.join(values + [text])
    if block not in funcs:
        funcs.append(block)
    return Code(fn['fn'])


def with_functions(obj, funcs, left):
    """ The model spec with each function written as the name of its definition; see `func_code()` """
    if isinstance(obj, dict):
        if 'fn' in obj and set(obj) <= {'fn', 'source', 'values'}:
            return func_code(obj, funcs, left)
        return {k:with_functions(v, funcs, left) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [with_functions(v, funcs, left) for v in obj]
    return obj


def operand(text):
    """ A rate after @, in brackets unless it is a product of simple terms (as in S >> I @ beta * I) """
    node = ast.parse(text, mode='eval').body
    def simple(n):
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Div)):
            return simple(n.left) and simple(n.right)
        return isinstance(n, (ast.Name, ast.Constant, ast.Call, ast.Attribute))
    return text if simple(node) else f'({text})'


def unitless_dur(val):
    """ Whether a value in the model spec is a duration with no unit, which is written as a plain number """
    return isinstance(val, dict) and val.get('type') == 'dur' and val['unit'] is None


def disease_code(spec, lines, left):
    """ The Python for one declared disease; returns the name of its variable """
    if spec['type'] != 'Disease':
        unknown(f'a module of type {spec["type"]}')
    if spec['functions']:
        unknown(f'functions ({sc.strjoin(f["name"] for f in spec["functions"])})')
    name = spec['name']
    add = lines.append

    # Parameters
    pars = {p['name']:p for p in spec['pars']}
    if pars:
        add('# Parameters')
    for p in spec['pars']:
        extra = ''.join(f', {k}={p[k]!r}' for k in ['bounds', 'source', 'doc'] if p[k] is not None)
        if isinstance(p['prior'], dict): # A distribution
            extra += f', prior={code_value(p["prior"])}'
        elif p['prior'] is not None:
            left.append(f'the prior of {p["name"]} ({p["prior"]})')
        add(f'{p["name"]} = ss.par({code_value(p["value"])}{extra})')
    if pars:
        add('')

    # States
    add('# States')
    plain = [] # A run of states with nothing but a name, declared together
    def flush():
        if len(plain) > 1:
            add(f"{', '.join(plain)} = ss.states('{' '.join(plain)}')")
        elif plain:
            add(f"{plain[0]} = ss.State('{plain[0]}')")
        plain.clear()
    linked = {spec.get('first') or spec['states'][0]['name']} if spec['states'] else set() # The states Starsim takes to be people: those connected by transitions to the first
    for i in range(len(spec['states'])):
        for t in spec['transitions']:
            ends = set(t['src'] or []) | set(t['dst'] or [])
            if t['src'] and t['dst'] and ends & linked:
                linked |= ends
    for s in spec['states']:
        guessed = f'states.{s["name"]}.people' in spec['guessed']
        extra = dict(people=False) if not s['people'] and not guessed else {}
        if s['people'] and not guessed and s['name'] not in linked: # People that Starsim would not otherwise take to be
            extra['people'] = True
        extra.update({k:s[k] for k in ['rel_sus', 'rel_trans'] if s[k] != 1.0})
        if s['label'] is not None:
            extra['label'] = s['label']
        if isinstance(s['init'], str):
            extra['init'] = s['init'] # An expression of the parameters: written as Python below
        if extra:
            flush()
            args = ', '.join(f'{k}={code_expr(v) if k == "init" else repr(v)}' for k,v in extra.items())
            add(f"{s['name']} = ss.State('{s['name']}', {args})")
        else:
            plain.append(s['name'])
    flush()
    add('')

    # Transitions
    def endpoint(names):
        if not names:
            return 'ss.null'
        return names[0] if len(names) == 1 else '[' + ', '.join(names) + ']'
    members = [] # Transitions and outcomes, written as variables (named from them), so that outcomes and results can use them
    kwargs = []
    for t in spec['transitions']:
        rate, dur = code_expr(t['rate']), None
        d = t['dur']
        if d is not None:
            if d['dist'] == 'expon' and isinstance(d.get('mean'), list):
                dur = d['mean'][1]
            elif d['dist'] == 'expon' and 'rate' in d:
                rate = d['rate'][1]
            else:
                held = [p for p in spec['pars'] if p['value'] == d]
                dur = held[0]['name'] if held else code_value(d)
        src, dst = endpoint(t['src']), endpoint(t['dst'])
        untyped = dur in pars and unitless_dur(pars[dur]['value']) # Written as a number, which @ would read as a rate
        simple = not untyped and len(t['src'] or []) <= 1 and len(t['dst'] or []) <= 1 and t['p'] is None and t['stages'] is None and t['label'] is None and not t.get('migration') and t.get('after') is None and t.get('before') is None
        if simple:
            members.append((t['name'], f'{src} >> {dst} @ {operand(rate if rate is not None else dur)}'))
        else:
            args = [src, dst]
            if rate is not None:
                args.append(f'rate={rate}')
            if dur is not None:
                args.append(f'dur={dur}')
            if t['p'] is not None:
                args.append('p=[' + ', '.join('None' if p is None else code_expr(p) for p in t['p']) + ']')
            for k in ['stages', 'label']:
                if t[k] is not None:
                    args.append(f'{k}={t[k]!r}')
            for k in ['after', 'before']:
                if t.get(k) is not None:
                    args.append(f'{k}={code_value(t[k])}')
            members.append((t['name'], f'ss.Transition({", ".join(args)})'))
        if t.get('migration'):
            unknown(f'the migration of {t["name"]}')

    # Results and outcomes
    for r in spec['results']:
        if r['type'] == 'result':
            kwargs.append((r['name'], code_expr(r['source'])))
        else:
            extra = ''.join(f', {k}={code_value(r[k])}' for k in ['delay', 'duration'] if r[k] is not None)
            lik = r['likelihood']
            if isinstance(lik, dict): # Its distribution, with each parameter a model parameter (by name) or a number
                args = ', '.join(f'{k}={v["par"] if isinstance(v, dict) else repr(v)}' for k,v in lik['pars'].items())
                extra += f', likelihood=ss.{lik["type"]}({args})'
            elif lik is not None:
                left.append(f'the likelihood of {r["name"]} ({lik})')
            source = r['source']
            members.append((r['name'], f'ss.outcome({code_expr(source)}{extra})'))

    for u in spec.get('updates', []): # Counts set each step
        members.append((u['name'], f"ss.update({u['state']}, {code_expr(u['value'])})"))

    # Initial values
    inits = {s['name']:s['init'] for s in spec['states'] if s['init'] is not None and not isinstance(s['init'], str)}
    if inits:
        kwargs.append(('init', 'dict(' + ', '.join(f'{k}={code_value(v)}' for k,v in inits.items()) + ')'))
    if spec['init_prev'] and 'init_prev' in pars: # A parameter, e.g. of ss.SIR
        kwargs.append(('init_prev', 'init_prev'))
    elif spec['init_prev']:
        kwargs.append(('init_prev', code_value(spec['init_prev'])))
    if spec.get('first') and any(s['people'] for s in spec['states']): # Written, so that the model does not depend on the rule that finds it
        kwargs.append(('first', repr(spec['first'])))
    kwargs.append(('name', repr(name)))

    var = name if name.isidentifier() else 'disease'
    members = ordered(members)
    if members:
        add('# Transitions and outcomes')
        w = max(len(k) for k,v in members)
        for k,v in members:
            add(f'{k:{w}s} = {v}')
        add('')
    width = max(len(k) for k,v in kwargs)
    used = {n for t in spec['transitions'] for n in (t['src'] or []) + (t['dst'] or [])}
    unused = [s['name'] for s in spec['states'] if s['name'] not in used] # Found only if given to the disease
    add(f'{var} = ss.Disease(')
    args = unused + [k for k,v in members]
    for i in range(0, len(args), 6):
        add(f'    {", ".join(args[i:i+6])},')
    for k,v in kwargs:
        add(f'    {k:{width}s} = {v},')
    add(')')
    transfers = spec.get('transfers', [])
    dims = [strat['name'] for strat in spec['stratifications']]
    for strat in spec['stratifications']: # Each with the transfers along it
        own = {tf['name']:Code(transfer_code(tf)) for tf in transfers if tf['dim'] == strat['name']}
        add(f'{var}.stratify({kwargs_code({**movable(strat, left), **own})})')
    rest = {tf['name']:Code(transfer_code(tf)) for tf in transfers if tf['dim'] not in dims}
    if rest: # Along a dimension of the sim
        add(f'{var}.stratify({kwargs_code(rest)})')
    add('')
    return var


def transfer_code(tf):
    """ The Python for a transfer in the model spec """
    target = tf['dst'] if tf['src'] is None else f"{tf['src']} -> {tf['dst']}"
    args = [f"{tf['dim']}={target!r}"] + [f'{k}={code_expr(tf[k])}' for k in ['rate', 'dur'] if tf[k] is not None]
    args += [f'{k}={tf[k]!r}' for k in ['states', 'label'] if tf[k] is not None]
    return f"ss.transfer({', '.join(args)})"


def contacts_code(net):
    """ A contact matrix as the call that makes it, or a named one as the call that looks it up (or None if the network is not one) """
    if net.get('contacts') is not None:
        pars = net['pars']
        if pars.get('beta') != 1.0 or pars.get('diseases') is not None: # Not arguments of ss.contacts()
            return None
        mats = net['contacts']
        args = [f'{mats["matrix"]!r}, bands={mats["bands"]!r}'] if 'bands' in mats else [f'{dim}={mat!r}' for dim,mat in mats.items()]
        if net.get('name') != 'contacts':
            args.append(f'name={net["name"]!r}')
        return f"ss.contacts({', '.join(args)})"
    ref = net.get('data')
    if not ref or ref.get('layer') is None:
        return None
    args = named_args(ref, contacts_keys)
    if args.get('source') == 'prem2021':
        args.pop('source')
    if args.get('layer') == 'all':
        args.pop('layer')
    if net.get('name') not in [None, ref.get('layer') if ref.get('layer') != 'all' else 'contacts']:
        args['name'] = net['name']
    return f"ss.contacts({ref['place']!r}{''.join(f', {k}={v!r}' for k,v in args.items())})"


def movable(strat, left):
    """ The arguments to stratify(), with the movement between places written as ss.Mobility() where it is a matrix """
    mob = strat['mobility']
    plain = {k:v for k,v in strat.items() if k not in ['geo', 'split_from']}
    out = strat_kwargs({**plain, 'mobility':None})
    if strat.get('geo'): # A named geography as the call that looks it up, with its own split
        ref = strat['geo']
        out[strat['name']] = Code(f"ss.geo({ref['place']!r}{''.join(f', {k}={v!r}' for k,v in named_args(ref, geo_keys).items())})")
        out.pop('split')
    if strat.get('split_from'):
        out.pop('split', None)
    if mob is None:
        return out
    if isinstance(mob, dict) and mob.get('matrix') is not None and not mob.get('data'):
        args = ', '.join(f'{k}={v!r}' for k,v in mob.items() if v is not None)
        out['mobility'] = Code(f'ss.Mobility({args})')
    else:
        left.append(f'the movement between places of "{strat["name"]}": give it with mobility=ss.Mobility(...) in stratify()')
    return out


def ordered(members):
    """ Transitions and outcomes, each after the ones it uses (the spec lists them by name) """
    names = {k for k,v in members}
    deps = {k:{n.id for n in ast.walk(ast.parse(v, mode='eval')) if isinstance(n, ast.Name)} & names - {k} for k,v in members}
    out, done = [], set()
    while len(out) < len(members):
        ready = [(k,v) for k,v in members if k not in done and deps[k] <= done]
        if not ready: # A cycle: keep the rest in order
            ready = [(k,v) for k,v in members if k not in done]
        for k,v in ready:
            out.append((k, v))
            done.add(k)
    return out


def kwargs_code(kwargs):
    """ Keyword arguments as Python, leaving out those that are None """
    return ', '.join(f'{k}={code_value(v)}' for k,v in kwargs.items() if v is not None)


def modify_code(mod, mods):
    """ An ss.modify as Python, with a function in it as the name of its definition (see `with_functions()`), and a condition as in `cond_code()` """
    write = lambda val: str(val) if isinstance(val, Code) else repr(val)
    mod = dict(mod, **{k:Code(time_code(mod[k])) for k in ['start', 'stop'] if mod.get(k) is not None}) # Written with str()
    if isinstance(mod.get('where'), str) and not isinstance(mod['where'], Code):
        mod['where'] = Code(cond_code(mod['where'], mods))
    keys = ['multiply', 'add', 'set', 'start', 'stop', 'when', 'where']
    args = [repr(mod['target'])] + [f'{k}={write(mod[k])}' for k in keys if mod.get(k) is not None]
    return f"ss.modify({', '.join(args)}, name={mod['name']!r})"


delivery_types = ['Delivery', 'vaccinate', 'treat', 'screen']
delivery_keys = ['at_age', 'start', 'stop', 'at', 'interval', 'where', 'coverage', 'capacity', 'on', 'when', 'deliver', 'trace', 'on_positive', 'on_negative', 'disease']
cond_keys = ['where', 'on', 'detects']


def owners(spec):
    """ The variable of each declared disease in the code (see `disease_code()`), with the names in it that a condition can use: its states, transitions, and stratifications """
    out = {}
    for mod in spec['modules']:
        var = mod['name'] if mod['name'].isidentifier() else 'disease'
        out[mod['name']] = (var, {s['name'] for s in mod['states']} | {t['name'] for t in mod['transitions']} | {g['name'] for g in mod['stratifications']})
    return out


def cond_code(text, mods):
    """ A condition written as text in the model spec (e.g. "(people.age < 5)") as Python: an attribute of people from ss.People, and a state, transition, or stratification from its disease """
    class Names(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'people':
                return ast.Attribute(ast.Attribute(ast.Name('ss'), 'People'), node.attr)
            elif isinstance(node.value, ast.Name) and node.value.id in mods:
                return ast.Attribute(ast.Name(mods[node.value.id][0]), node.attr)
            return self.generic_visit(node)
        def visit_Name(self, node):
            having = [var for var,names in mods.values() if node.id in names]
            if having:
                return ast.Attribute(ast.Name(having[0]), node.id)
            return ast.parse(f"ss.conditions.Cond('dim', {node.id!r})", mode='eval').body # A stratification of the sim
    return ast.unparse(Names().visit(ast.parse(text, mode='eval').body))


def delivery_code(iv, byname, mods):
    """ A delivery in the model spec as Python, with its product, and with each delivery that follows it written in it """
    def value(key, val):
        if key in cond_keys and isinstance(val, str) and not isinstance(val, Code):
            return cond_code(val, mods)
        elif key in ['on_positive', 'on_negative'] and val in byname:
            return delivery_code(byname[val], byname, mods)
        return code_value(val)
    args = []
    prod = iv.get('product_spec')
    if prod is not None:
        kwargs = [f'{k}={value(k, v)}' for k,v in prod.items() if k not in ['type', 'name']]
        kwargs.append(f'name={prod["name"]!r}')
        args.append(f"ss.{prod['type']}({', '.join(kwargs)})")
    if iv.get('move') is not None:
        args.append(f'move={iv["move"]!r}')
    args += [f'{k}={value(k, iv[k])}' for k in delivery_keys if iv.get(k) is not None]
    args.append(f'name={iv["name"]!r}')
    return f"ss.{iv['type']}({', '.join(args)})"


def to_code(spec, filename=None, header=None, footer=None, code=None, missing=None):
    """
    Write a model spec as a Python file that builds the same sim

    The diseases are written in the operator form (`S >> I @ beta * I`), with each
    parameter and state declared first, and each function used in the model defined
    before them (from its source). What cannot be written (networks other than contact
    matrices, interventions other than `ss.modify` and the deliveries, movement between places
    from data, likelihoods other than a distribution, and functions whose source is not a plain definition) is listed
    in comments at the end, and in a warning (`ss.LossyExport`), since the code then
    builds a different model until it is added by hand.

    Args:
        spec (dict/str): the spec (from `sim.to_spec()`), or a JSON file holding one
        filename (str): if given, write the code to this file
        header (str): a docstring for the top of the file
        footer (list): comments to add at the end, one line each (e.g. what an import did not translate)
        code (dict): Python for arguments to `ss.Sim` that the spec does not hold as data, e.g. `dict(networks='ss.contacts(...)')`
        missing (list): parts of the sim that are not in the spec, to list with what is not written (e.g. the data)

    Returns:
        The code, as a string
    """
    if not isinstance(spec, dict):
        spec = sc.loadjson(spec)
    left = list(missing or []) # What is not written, listed at the end rather than failing (as for an export)
    funcs = [] # The definitions of the functions used in the model
    spec = dict(spec, modules=with_functions(spec['modules'], funcs, left))
    code = dict(code or {})
    extra = {}
    for key,simkey in [('networks', 'networks'), ('modifiers', 'connectors'), ('interventions', 'interventions')]:
        if key in code or simkey in code:
            extra[simkey] = code.get(simkey, code.get(key))
            continue
        mods = with_functions(spec.get(key) or [], funcs, left)
        byname = {m.get('name'):m for m in mods}
        followups = {m[k] for m in mods for k in ['on_positive', 'on_negative'] if isinstance(m.get(k), str)} # Written in the delivery they follow
        written = []
        for mod in mods: # In the order given
            if mod.get('type') == 'modify':
                written.append(modify_code(mod, owners(spec)))
            elif contacts_code(mod):
                written.append(contacts_code(mod))
            elif mod.get('type') in delivery_types:
                if mod['name'] not in followups:
                    written.append(delivery_code(mod, byname, owners(spec)))
            else:
                left.append(f'{key[:-1]} "{mod.get("name", "?")}" ({mod.get("type", "?")}): add it to ss.Sim({simkey}=...)')
        if written:
            extra[simkey] = '[' + ', '.join(written) + ']'
    lines = []
    if header:
        lines += ['"""', *header.strip().splitlines(), '"""']
    lines += ['import starsim as ss', '']
    if funcs:
        lines += ['# Functions'] + [line for block in funcs for line in block.splitlines() + ['']]
    names = [disease_code(mod, lines, left) for mod in spec['modules']]
    cfg = spec['config']
    start, stop = cfg['start'], cfg['stop']
    unit = cfg['unit']
    kwargs = dict(solver=cfg['solver'] if cfg['solver'] != 'auto' else None, integrator=cfg['integrator'], approx=cfg['approx'], unit=unit)
    if unit is not None and sc.isnumber(start) and sc.isnumber(stop) and not start: # No dates: durations in the sim's unit
        kwargs['stop'] = dict(type='dur', value=stop, unit=unit)
    else:
        kwargs.update(start=start, stop=stop)
    kwargs.update(dt=cfg['dt'] if cfg['dt'] != 1 else None, total_pop=cfg['total_pop'], n_agents=cfg['n_agents'], rand_seed=cfg['rand_seed'] if cfg['rand_seed'] != 1 else None, substeps=cfg.get('substeps'), use_aging=cfg.get('use_aging'))
    if cfg.get('people'): # A named population as the call that looks it up
        ref = cfg['people']
        kwargs['people'] = Code(f"ss.people({ref['place']!r}{''.join(f', {k}={v!r}' for k,v in named_args(ref, people_keys).items())})")
        if np.isclose(kwargs['total_pop'], ref['total']):
            kwargs['total_pop'] = None
    kwargs.update({k:Code(v) for k,v in extra.items()})
    for key in ['total_pop', 'n_agents', 'dt']:
        if sc.isnumber(kwargs[key]) and float(kwargs[key]).is_integer():
            kwargs[key] = int(kwargs[key])
    diseases = names[0] if len(names) == 1 else '[' + ', '.join(names) + ']'
    lines.append(f'sim = ss.Sim({diseases}, {kwargs_code(kwargs)})')
    for strat in spec['stratify']:
        lines.append(f'sim.stratify({kwargs_code(movable(strat, left))})')
    footer = list(footer or [])
    if left:
        footer += ['Not written as code (LossyExport; add by hand):'] + [f'  {item}' for item in left]
        ss.warn(f'the code builds a different model until these are added by hand: {"; ".join(left)}', ss.LossyExport)
    if footer:
        lines += [''] + [f'# {line}' for line in footer]
    lines += ['', "if __name__ == '__main__':", '    sim.run()', '    sim.plot()', '']
    code = '\n'.join(lines)
    if any('np.' in line for line in lines if not line.startswith('#')):
        code = code.replace('import starsim as ss\n', 'import numpy as np\nimport starsim as ss\n', 1)
    if filename is not None:
        sc.savetext(filename, code)
    return code


#%% Comparing two models

def shown(val, unit=None):
    """ A value from the model spec, short and in the sim's unit where it has one, e.g. "0.5/day" or "lognorm_ex(mean=10 day, std=3 day)" """
    if isinstance(val, dict) and val.get('type') in ['dur', 'per', 'prob', 'freq'] and val.get('unit'):
        num, u = val['value'], val['unit']
        if unit and u != unit:
            factor = ss.time.get_dur_class(u)(1).to(unit).value
            num, u = (num*factor if val['type'] == 'dur' else num/factor), unit
        num = f'{num:.6g}'
        return f'{num} {u}' if val['type'] == 'dur' else f'{num}/{u}' if val['type'] == 'per' else f'{val["type"]} {num}/{u}'
    elif isinstance(val, dict) and 'dist' in val:
        if val['dist'] == 'expon' and isinstance(val.get('mean', val.get('rate')), list):
            return 'exponential'
        return f'{val["dist"]}(' + ', '.join(f'{k}={shown(v, unit)}' for k,v in val.items() if k != 'dist') + ')'
    elif isinstance(val, dict) and 'type' in val:
        return f'{val["value"]:.6g}' if sc.isnumber(val['value']) else str(val['value'])
    elif isinstance(val, float):
        return f'{val:.6g}'
    return str(val)


def resolved(t, pars, unit):
    """ A transition's rate or dwell time with its parameters written in as values, so that two models can be compared whatever their parameters are called """
    def sub(text):
        if text is None:
            return None
        node = ast.parse(text, mode='eval').body
        for n in ast.walk(node):
            for field,val in ast.iter_fields(n):
                if isinstance(val, ast.Name) and val.id in pars:
                    setattr(n, field, ast.Name(shown(pars[val.id], unit)))
        if isinstance(node, ast.Name) and node.id in pars:
            return shown(pars[node.id], unit)
        return ast.unparse(node)
    d = t['dur']
    if d is not None and d.get('dist') == 'expon' and isinstance(d.get('rate'), list):
        return f'rate {shown(pars.get(d["rate"][1]), unit)}'
    elif d is not None and d.get('dist') == 'expon' and isinstance(d.get('mean'), list):
        return f'dwell exponential, mean {shown(pars.get(d["mean"][1]), unit)}'
    elif d is not None:
        held = [k for k,v in pars.items() if v == d]
        return f'dwell {shown(d, unit)}' if not held else f'dwell {shown(pars[held[0]], unit)}'
    rate = sub(t['rate'])
    return f'rate {rate}' + (f', p={[sub(p) for p in t["p"]]}' if t['p'] else '')


def refs(t):
    """ The names a transition's rate, branch probabilities, and dwell time use """
    out = set()
    for text in [t['rate']] + list(t['p'] or []):
        if text is not None:
            out |= {n.id for n in ast.walk(ast.parse(text, mode='eval')) if isinstance(n, ast.Name)}
    d = t['dur'] or {}
    out |= {v[1] for v in d.values() if isinstance(v, list) and len(v) == 2 and v[0] == 'par'}
    return out


def diff(a, b, output=False):
    """
    Compare two models in their own terms: states, transitions, parameters, run settings, and interventions

    The four lines are the four hashed parts of `sim.hash()`. Transitions are matched
    by name, then by their source and destination, and compared with their parameters
    written in as values, so a model and a translation of it whose parameters or
    transitions have other names are reported as the same where they are.

    Args:
        a (Sim/dict/str): a sim, a model spec, or a JSON file holding one
        b (Sim/dict/str): the other
        output (bool): if True, return the report rather than printing it

    **Example**:

        my_sim.to_spec('model.json') # Saved earlier
        ss.diff('model.json', my_sim)
    """
    specs = []
    for x in [a, b]:
        if isinstance(x, ss.Sim):
            specs.append(x.to_spec())
        elif isinstance(x, dict):
            specs.append(x)
        else:
            specs.append(sc.loadjson(x))
    sa, sb = specs
    unit = sa['config'].get('unit') or sb['config'].get('unit')
    model, pars = [], []

    # Modules, matched by name, or in order if the names differ
    ma = {m['name']:m for m in sa['modules']}
    mb = {m['name']:m for m in sb['modules']}
    if set(ma) != set(mb) and len(ma) == len(mb):
        pairs = list(zip(ma.values(), mb.values()))
    else:
        pairs = [(ma[k], mb[k]) for k in ma if k in mb]
        model += [f"module '{k}' removed" for k in ma if k not in mb] + [f"module '{k}' added" for k in mb if k not in ma]
    for x,y in pairs:
        prefix = f"{x['name']}: " if len(pairs) > 1 else ''
        px = {p['name']:p['value'] for p in x.get('pars', [])}
        py = {p['name']:p['value'] for p in y.get('pars', [])}

        # States
        sx, sy = [s['name'] for s in x.get('states', [])], [s['name'] for s in y.get('states', [])]
        model += [f"{prefix}state '{s}' removed" for s in sx if s not in sy] + [f"{prefix}state '{s}' added" for s in sy if s not in sx]
        first = x.get('first', sx[0] if sx else None)
        if x.get('first') != y.get('first') and None not in (x.get('first'), y.get('first')):
            model.append(f"{prefix}first {x['first']} -> {y['first']}")
        for s in [s for s in x.get('states', []) if s['name'] in sy]:
            t = next(o for o in y['states'] if o['name'] == s['name'])
            for key in ['people', 'rel_sus', 'rel_trans', 'init']:
                rest = key == 'init' and s['name'] == first and None in (s[key], t[key]) # The first state starts with the rest, whether or not it is written
                if s[key] != t[key] and not rest:
                    (model if key == 'people' else pars).append(f"{prefix}{s['name']}.{key} {shown(s[key])} -> {shown(t[key])}")

        # Transitions: by name, then by endpoints
        tx = {t['name']:t for t in x.get('transitions', [])}
        ty = {t['name']:t for t in y.get('transitions', [])}
        ends = lambda t: (tuple(t['src'] or []), tuple(t['dst'] or []))
        matched = [(k, k) for k in tx if k in ty]
        left = [k for k in tx if k not in ty]
        right = [k for k in ty if k not in tx]
        for k in list(left):
            same = [j for j in right if ends(ty[j]) == ends(tx[k])]
            if len(same) == 1:
                matched.append((k, same[0]))
                left.remove(k)
                right.remove(same[0])
        model += [f"{prefix}transition '{k}' removed" for k in left] + [f"{prefix}transition '{k}' added ({sc.strjoin(ty[k]['src'] or ['in'])} -> {sc.strjoin(ty[k]['dst'] or ['out'])})" for k in right]
        for k,j in matched:
            t, u = tx[k], ty[j]
            name = f"'{k}'" if k == j else f"'{k}' (called '{j}')"
            if ends(t) != ends(u):
                model.append(f'{prefix}transition {name} {ends(t)} -> {ends(u)}')
            ra, rb = resolved(t, px, unit), resolved(u, py, unit)
            if ra != rb:
                same_form = t['rate'] == u['rate'] and t['p'] == u['p'] and (t['dur'] or {}).get('dist') == (u['dur'] or {}).get('dist') and refs(t) == refs(u)
                if not same_form: # Otherwise the parameters' values say what changed
                    model.append(f'{prefix}transition {name} {ra} -> {rb}')
            elif k != j:
                model.append(f'{prefix}transition {name}: the same')

        # Parameters with the same name and the same use, results, and stratifications
        names = dict(matched)
        use_x = {p:{names[k] for k,t in tx.items() if p in refs(t) and k in names} for p in px}
        use_y = {p:{j for j,t in ty.items() if p in refs(t)} for p in py}
        for k in [k for k in px if k in py and px[k] != py[k] and use_x[k] == use_y[k]]:
            pars.append(f'{prefix}{k} {shown(px[k], unit)} -> {shown(py[k], unit)}')
        rx = {r['name']:r for r in x.get('results', [])}
        ry = {r['name']:r for r in y.get('results', [])}
        model += [f"{prefix}result '{k}' removed" for k in rx if k not in ry] + [f"{prefix}result '{k}' added" for k in ry if k not in rx]
        model += [f"{prefix}result '{k}' {rx[k].get('source')} -> {ry[k].get('source')}" for k in rx if k in ry and rx[k] != ry[k]]
        gx = {g['name']:g for g in x.get('stratifications', [])}
        gy = {g['name']:g for g in y.get('stratifications', [])}
        model += [f"{prefix}stratification '{k}' removed" for k in gx if k not in gy] + [f"{prefix}stratification '{k}' added" for k in gy if k not in gx]
        model += [f"{prefix}stratification '{k}' changed" for k in gx if k in gy and gx[k] != gy[k]]

    # Networks, run settings, and interventions
    for key in ['networks', 'stratify', 'modifiers']:
        if sa.get(key) != sb.get(key):
            model.append(f'{key}: {len(sa.get(key) or [])} -> {len(sb.get(key) or [])}' if len(sa.get(key) or []) != len(sb.get(key) or []) else f'{key} changed')
    ca, cb = sa['config'], sb['config']
    config = [f'{k} {shown(ca.get(k))} -> {shown(cb.get(k))}' for k in ca if k != 'rand_seed' and ca.get(k) != cb.get(k)]
    ia, ib = sa.get('interventions') or [], sb.get('interventions') or []
    names = lambda iv: sc.strjoin(i.get('name', '?') for i in iv) or 'none'
    scenario = [] if ia == ib else [f'{names(ia)} -> {names(ib)}']

    lines = [f'{key + ":":9s} {"; ".join(items) if items else "same"}' for key,items in [('model', model), ('pars', pars), ('config', config), ('scenario', scenario)]]
    text = '\n'.join(lines)
    return text if output else print(text)
