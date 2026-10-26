"""
The model spec (written by `sim.to_spec()`, here, for `sim.hash()` and `ss.diff()`) as code (`ss.to_code()`, for the declared parts of a model), and the parser of expressions in it
"""
import ast
import keyword
import textwrap
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['to_code', 'diff']

# Typed quantities that can be written inside a rate, e.g. perday(0.5)
time_names = ['days', 'weeks', 'months', 'years', 'perday', 'perweek', 'permonth', 'peryear',
              'probperday', 'probperweek', 'probpermonth', 'probperyear', 'freqperday', 'freqperweek', 'freqpermonth', 'freqperyear']

# Functions that can be written inside a rate, by their name in the spec
func_names = ['exp', 'log', 'sqrt', 'absolute', 'sin', 'cos', 'expit', 'minimum', 'maximum']
comparisons = {ast.Lt:'<', ast.LtE:'<=', ast.Gt:'>', ast.GtE:'>='}


def unknown(what):
    """ Refuse to write a part of a declared module that to_code() does not write, rather than write a different model """
    errormsg = f'ss.to_code() cannot write {what}: only the declared parts of a model are written as code. Save the sim with ss.save("run.sim", sim) instead'
    raise NotImplementedError(errormsg)


class Parser:
    """ Read a rate or a result written as text, e.g. "beta * W.n / (W.n + K)", allowing only the names the model declares, and people's properties (people.age) """
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
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'people':
            return getattr(ss.ppl, node.attr)
        elif isinstance(node, ast.Attribute) and node.attr in ['n', 'frac', 'total', 'dwell']:
            return getattr(ev(node.value), node.attr)
        elif isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            ops = {ast.Add:lambda: a + b, ast.Sub:lambda: a - b, ast.Mult:lambda: a * b, ast.Div:lambda: a / b, ast.Pow:lambda: a ** b, ast.BitAnd:lambda: a & b, ast.BitOr:lambda: a | b}
            if type(node.op) in ops:
                return ops[type(node.op)]()
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return ~ev(node.operand)
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in comparisons:
            a, b = ev(node.left), ev(node.comparators[0])
            a = a if isinstance(a, ss.conditions.Cond) else ss.transitions.as_expr(a) # A condition on people's properties stays one
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
        levels = ss.geo.geography(strat['geo']['place'], **named_args(strat['geo'], geo_keys))
    if strat['kind'] == 'clock' and strat.get('unit'): # Durations
        levels = Code(f"ss.{strat['unit']}s({levels})")
    kind = strat['kind']
    implied = 'age' if strat['name'] == 'age' else 'plain' # The kind stratify() gives when none is given
    out = {strat['name']:levels, 'split':split, 'adjust':strat['adjust'] or None, 'kind':None if (kind == implied and not strat.get('geo')) else kind}
    if strat.get('split_from'): # Taken from the named population again
        out.pop('split')
    elif isinstance(levels, list) and isinstance(split, list) and len(levels) == len(split) and kind != 'clock': # Each level with its share, e.g. age={0: 0.3, 15: 0.6, 65: 0.1}
        out[strat['name']] = {int(k) if sc.isnumber(k) and not isinstance(k, bool) and float(k).is_integer() else k: v for k,v in zip(levels, split)}
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
    """ The Python for a value in the model spec """
    if isinstance(val, Code):
        return str(val)
    elif isinstance(val, ss.by) and val.array is not None:
        return f'ss.by({np.asarray(val.array).tolist()!r}, dims={list(val.dims)!r})'
    elif isinstance(val, dict):
        if val.get('type') == 'peract':
            items = {k:v for k,v in val.items() if k != 'type'}
            return f'ss.peract({code_value(items["value"])})' if 'value' in items else 'ss.peract(' + ', '.join(f'{k}={code_value(v)}' for k,v in items.items()) + ')'
        elif 'type' in val:
            unit = val['unit']
            if unit is None:
                return code_value(val['value'])
            if val['type'] == 'dur':
                return f'ss.{unit}s({code_value(val["value"])})'
            prefix = '' if val['type'] == 'per' else val['type']
            return f'ss.{prefix}per{unit}({code_value(val["value"])})'
        elif 'dist' in val:
            args = ', '.join(f'{k}={code_value(v)}' for k,v in val.items() if k != 'dist')
            prefix = 'ss.' if is_dist(val['dist']) else '' # A distribution of the user's own is written by its name: see with_functions()
            return f'{prefix}{val["dist"]}({args})'
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
        elif 'keyed' in val:
            items = val['keyed'].items()
            if all(str(k).isidentifier() and not keyword.iskeyword(str(k)) for k,v in items):
                return 'dict(' + ', '.join(f'{k}={code_value(v)}' for k,v in items) + ')'
            return '{' + ', '.join(f'{k!r}: {code_value(v)}' for k,v in items) + '}'
        elif 'seasonal' in val:
            f = val['seasonal']
            return f'ss.seasonal(mean={code_value(f["mean"])}, amplitude={f["amplitude"]!r}, peak={f["peak"]!r})'
        return '{' + ', '.join(f'{k!r}: {code_value(v)}' for k,v in val.items()) + '}' # A plain dict, e.g. adjust=
    elif isinstance(val, list):
        return '[' + ', '.join(code_value(v) for v in val) + ']'
    elif isinstance(val, float) and val.is_integer() and abs(val) < 1e15: # A whole number as an integer: 8, not 8.0
        return repr(int(val))
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
    """ The Python for a rate or result in the model spec: functions from NumPy (and SciPy's expit), typed quantities from Starsim, and people's properties as ss.ppl """
    if text is None:
        return None
    node = ast.parse(text, mode='eval').body
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'people':
            n.value = ast.Attribute(ast.Name('ss'), 'ppl')
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id == 'expit': # Imported from scipy.special: see to_code()
                continue
            elif n.func.id in func_names:
                n.func = ast.Attribute(ast.Name('np'), n.func.id)
            elif n.func.id in time_names or hasattr(ss, n.func.id): # Every Starsim name, e.g. ss.by() or ss.seasonal()
                n.func = ast.Attribute(ast.Name('ss'), n.func.id)
    return ast.unparse(node)


def func_code(fn, funcs, left):
    """ A function in the model spec as the name of its definition, which is added to funcs (with the values it closes over); listed in left if its source is not a plain definition, and then written by its name (so the code fails until it is defined, rather than run without it), or as None if it has none (a lambda) """
    source = fn.get('source')
    text = textwrap.dedent(source).strip() if source else ''
    if not text.startswith('def '):
        if fn['fn'].isidentifier():
            left.append(f'the function {fn["fn"]}, whose source is not a plain definition: define it before the code that uses it')
            return Code(fn['fn'])
        left.append(f'the function {fn["fn"]} (written as None): define it and pass it in place of None')
        return None
    values = [f'{k} = {v!r}' for k,v in (fn.get('values') or {}).items() if not k.startswith('default_')] # Defaults are in the source
    block = '\n'.join(values + [text])
    if block not in funcs:
        funcs.append(block)
    return Code(fn['fn'])


def is_dist(name):
    """ Whether a distribution in the model spec is one of Starsim's, e.g. "lognorm_ex" """
    cls = getattr(ss, name, None)
    return isinstance(cls, type) and issubclass(cls, ss.Dist)


def with_functions(obj, funcs, left):
    """ The model spec with each function written as the name of its definition (see `func_code()`); a distribution that is not Starsim's is listed in left """
    if isinstance(obj, dict):
        if 'fn' in obj and set(obj) <= {'fn', 'source', 'values', 'ref', 'version'}:
            return func_code(obj, funcs, left)
        if isinstance(obj.get('dist'), str) and not is_dist(obj['dist']):
            item = f'the distribution {obj["dist"]}, which is not part of Starsim (written as {obj["dist"]}(...)): define or import it before the code that uses it'
            if item not in left:
                left.append(item)
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
    for s in spec['states']:
        extra = dict(people=False) if not s['people'] else {} # A state is people unless declared a count
        extra.update({k:s[k] for k in ['rel_sus', 'rel_trans'] if s[k] != 1.0})
        if s['label'] is not None:
            extra['label'] = s['label']
        for key in ['init', 'init_n']:
            if isinstance(s.get(key), str):
                extra[key] = s[key] # An expression of the parameters: written as Python below
        if extra:
            flush()
            args = ', '.join(f'{k}={code_expr(v) if k in ["init", "init_n"] else repr(v)}' for k,v in extra.items())
            add(f"{s['name']} = ss.State('{s['name']}', {args})")
        else:
            plain.append(s['name'])
    flush()
    add('')

    # Strains, which the transitions can refer to (move=DS >> RIF)
    strained = [strat['strains'] for strat in spec['stratifications'] if strat.get('strains')] + ([spec['own_strains']] if spec.get('own_strains') else []) # Shared, or courses of their own from this natural history
    for st in strained:
        add('# Strains')
        lines += strains_lines(st, 'strains', left)
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
        simple = not untyped and len(t['src'] or []) <= 1 and len(t['dst'] or []) <= 1 and t['p'] is None and t['stages'] is None and t['label'] is None and not t.get('migration') and t.get('after') is None and t.get('before') is None and t.get('acquisition') is None and not t.get('move') and not t.get('increment') and not t.get('per')
        if simple:
            members.append((t['name'], f'{src} >> {dst} @ {operand(rate if rate is not None else dur)}'))
        else:
            args = [src, dst]
            if rate is not None:
                args.append(f'rate={rate}')
            if dur is not None:
                args.append(f'dur={dur}')
            if t.get('per'): # An outcome for each strain carried
                args += ["per='strain'"] + (['alone=True'] if t['per'].get('alone') else [f"p={t['per']['p']!r}"])
            elif t['p'] is not None:
                args.append('p=[' + ', '.join('None' if p is None else code_expr(p) for p in t['p']) + ']')
            for k in ['stages', 'label']:
                if t[k] is not None:
                    args.append(f'{k}={t[k]!r}')
            for k in ['after', 'before']:
                if t.get(k) is not None:
                    args.append(f'{k}={code_value(t[k])}')
            if t.get('acquisition') is not None:
                args.append(f'acquisition={t["acquisition"]}')
            if t.get('move'):
                args.append(move_code(t['move']))
            if t.get('increment'):
                args.append(f'increment={increment_code(t["increment"])}')
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
            if isinstance(lik, dict) and 'type' in lik: # Its distribution, with each parameter a model parameter (by name) or a number
                args = ', '.join(f'{k}={v["par"] if isinstance(v, dict) else repr(v)}' for k,v in lik['pars'].items())
                extra += f', likelihood=ss.{lik["type"]}({args})'
            elif lik is not None:
                left.append(f'the likelihood of {r["name"]} ({lik.get("fn", lik) if isinstance(lik, dict) else lik})')
            source = r['source']
            members.append((r['name'], f'ss.outcome({code_expr(source)}{extra})'))

    # Initial values
    for key in ['init', 'init_n']:
        inits = {s['name']:s[key] for s in spec['states'] if s.get(key) is not None and not isinstance(s[key], str)}
        if inits:
            kwargs.append((key, 'dict(' + ', '.join(f'{k}={code_value(v)}' for k,v in inits.items()) + ')'))
    if spec['init_prev'] and 'init_prev' in pars: # A parameter, e.g. of ss.SIR
        kwargs.append(('init_prev', 'init_prev'))
    elif spec['init_prev']:
        kwargs.append(('init_prev', code_value(spec['init_prev'])))
    people = [s['name'] for s in spec['states'] if s['people']]
    default = people and spec.get('first') == people[0] and people[0] in ['S', 'susceptible'] # The first state written, and named as the default is: left out
    if spec.get('first') and people and not default: # Otherwise written, so that the model does not depend on the rule that finds it
        kwargs.append(('first', repr(spec['first'])))
    kwargs.append(('name', repr(name)))
    if strained:
        kwargs.append(('strains', 'strains'))

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
    transfers = spec.get('transfers') or []
    dims = [strat['name'] for strat in spec['stratifications'] if not strat.get('strains')]
    for strat in spec['stratifications']: # Each with the transfers along it
        if strat['name'] in dims:
            own = {tf['name']:Code(transfer_code(tf)) for tf in transfers if tf['dim'] == strat['name']}
            add(f'{var}.stratify({kwargs_code({**movable(strat, left), **own})})')
    rest = {tf['name']:Code(transfer_code(tf)) for tf in transfers if tf['dim'] not in dims}
    if rest: # Along a dimension of the sim
        add(f'{var}.stratify({kwargs_code(rest)})')
    add('')
    return var


def strains_lines(spec, var, left, members=None):
    """ The Python that makes an `ss.Strains` from its model spec, as the variable `var` (with `members`, the variables of the diseases that are its strains) """
    values = spec.get('values') or {}
    first = '[' + ', '.join(members) + ']' if members is not None else repr(spec['strains'])
    kwargs = dict(shared=None if spec['shared'] is (members is None) else spec['shared'], max_strains=None if spec['max_strains'] == 1 else spec['max_strains'],
                  transmit=None if spec['transmit'] == 'one' else spec['transmit'], split=spec['split'], name=None if spec['name'] == 'strain' else spec['name'])
    kwargs.update(values)
    args = kwargs_code(kwargs)
    out = [f"{var} = ss.Strains({first}{', ' + args if args else ''})"]
    immunity = []
    for imm in spec['immunity']:
        due = imm['due_to']
        due = Code(f'{var}.{due.rpartition(".")[2]}') if isinstance(due, str) else Code(due[0]) if members is None and len(due) == 1 else due[0] if len(due) == 1 else due # The memory, or the state (a variable of the disease), or a name of each strain's state
        imm_kwargs = dict(due_to=due, scale=imm.get('scale'), rel_sus=imm.get('rel_sus'), rel_trans=imm.get('rel_trans'), dur=imm.get('dur'), p=imm.get('p'), combine=None if imm['combine'] == 'max' else imm['combine'])
        if imm.get('curve') is not None:
            left.append(f'the curve of an ss.immunity of the strains "{spec["name"]}" ({imm["curve"]})')
        immunity.append(f'ss.immunity({kwargs_code(imm_kwargs)})')
    if immunity:
        out.append(f"{var}.immunity = [{', '.join(immunity)}]")
    return out


def strains_code(spec, names, lines, left):
    """ The Python for a disease whose strains have courses of their own (`ss.strains.Family`), whose strains are written before it as diseases; returns the names of the sim's diseases, with the disease in place of its strains """
    members = spec['members']
    lines += ['# Strains with courses of their own'] + strains_lines(spec['strains'], f"{spec['name']}_strains", left, members=members)
    init_prev = f', init_prev={code_value(spec["init_prev"])}' if spec['init_prev'] is not None else ''
    lines += [f"{spec['name']} = ss.Disease(name={spec['name']!r}{init_prev}, strains={spec['name']}_strains)", '']
    return list(dict.fromkeys(spec['name'] if name in members else name for name in names)) # In the place of its first strain


def move_code(move, strains='strains'):
    """ The Python for move= in the model spec: levels by dimension, pairs of levels (from, to), or a change of strains (`DS >> RIF`, with `strains` the code for the disease's ss.Strains) """
    out = {}
    for dim,level in move.items():
        if isinstance(level, dict) and 'strains' in level: # A change of strains: the only move
            strain = lambda names: f"{strains}[{names[0]!r}]" if len(names) == 1 else '(' + ', '.join(f"{strains}[{n!r}]" for n in names) + ')'
            return f"move={strain(level['strains']['src'])} >> {strain(level['strains']['dst'])}"
        out[dim] = Code(repr((level['src'][0] if len(level['src']) == 1 else level['src'], level['dst']))) if isinstance(level, dict) and 'src' in level else level
    return 'move=dict(' + kwargs_code(out) + ')'


def increment_code(inc):
    """ The Python for an `increment=`: a name alone for adding 1 to one counter, else a dict """
    if len(inc) == 1 and list(inc.values())[0] == 1:
        return repr(list(inc)[0])
    return 'dict(' + ', '.join(f'{k}={v}' for k,v in inc.items()) + ')'


def transfer_code(tf):
    """ The Python for a transfer in the model spec """
    target = tf['dst'] if tf['src'] is None else (tf['src'], tf['dst'])
    first = [f'increment={increment_code({tf["dim"]:tf["increment"]})}'] if tf.get('increment') is not None else [f"{tf['dim']}={target!r}"]
    args = first + [f'{k}={code_expr(tf[k])}' for k in ['rate', 'dur'] if tf[k] is not None]
    args += [f"p=[{', '.join('None' if v is None else code_expr(v) for v in tf['p'])}]"] if tf.get('p') is not None else []
    args += [f'{k}={tf[k]!r}' for k in ['states', 'label'] if tf[k] is not None]
    return f"ss.transfer({', '.join(args)})"


def contacts_code(net):
    """ A contact matrix as the call that makes it, or a named one as the call that looks it up (or None if the network is not one) """
    if net.get('contacts') is not None:
        pars = net['pars']
        if pars.get('beta') != 1.0 or pars.get('diseases') is not None: # Not arguments of ss.contacts()
            return None
        mats = net['contacts']
        others = {dim:mat for dim,mat in mats.items() if dim not in ['ages', 'matrix', 'dims']} # A matrix for each of the model's stratifications
        if 'ages' in mats: # Its own age groups (with a matrix for each of other stratifications)
            matrix = f'dict(age={mats["matrix"]!r}, ' + ', '.join(f'{dim}={mat!r}' for dim,mat in others.items()) + ')' if others else repr(mats['matrix'])
            args = [f'{matrix}, ages={mats["ages"]!r}']
        elif 'dims' in mats:
            args = [f'{mats["matrix"]!r}, dims={mats["dims"]!r}']
        elif len(others) == 1:
            dim, mat = next(iter(others.items()))
            args = [repr(mat)] + ([f'dims={dim!r}'] if dim != 'age' else [])
        else:
            args = ['dict(' + ', '.join(f'{dim}={mat!r}' for dim,mat in others.items()) + ')']
        if net.get('name') != 'contacts':
            args.append(f'name={net["name"]!r}')
        if net.get('contacts_unit') is not None:
            args.append(f'unit={net["contacts_unit"]!r}')
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
    """ The arguments to stratify(), with the movement between places written as ss.geo.Mobility() where it is a matrix """
    mob = strat['mobility']
    plain = {k:v for k,v in strat.items() if k != 'geo'}
    out = strat_kwargs({**plain, 'mobility':None})
    if strat.get('geo'): # A named geography as the call that looks it up, with its own split
        ref = strat['geo']
        out[strat['name']] = Code(f"ss.geo.geography({ref['place']!r}{''.join(f', {k}={v!r}' for k,v in named_args(ref, geo_keys).items())})")
        out.pop('split', None)
    if strat.get('split_from'):
        out.pop('split', None)
    if mob is None:
        return out
    if isinstance(mob, dict) and mob.get('matrix') is not None and not mob.get('data'):
        args = ', '.join(f'{k}={v if k == "away" and isinstance(v, str) else repr(v)}' for k,v in mob.items() if v is not None) # A fitted share of time away by its parameter's name
        out['mobility'] = Code(f'ss.geo.Mobility({args})')
    else:
        left.append(f'the movement between places of "{strat["name"]}": give it with mobility=ss.geo.Mobility(...) in stratify()')
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
    write = lambda val: str(val) if isinstance(val, Code) else code_value(val) if isinstance(val, dict) else repr(val) # A share that varies in time is a value of the spec
    mod = dict(mod, **{k:Code(time_code(mod[k])) for k in ['start', 'stop'] if mod.get(k) is not None}) # Written with str()
    if isinstance(mod.get('where'), str) and not isinstance(mod['where'], Code):
        mod['where'] = Code(cond_code(mod['where'], mods))
    keys = ['multiply', 'add', 'set', 'efficacy', 'share', 'route', 'start', 'stop', 'when', 'where']
    args = [repr(mod['target'])] + [f'{k}={write(mod[k])}' for k in keys if mod.get(k) is not None]
    return f"ss.modify({', '.join(args)}, name={mod['name']!r})"


delivery_types = ['Delivery', 'vaccinate', 'treat', 'screen']
delivery_keys = ['strain', 'at_age', 'start', 'stop', 'at', 'interval', 'where', 'coverage', 'capacity', 'on', 'delay', 'when', 'deliver', 'trace', 'disease']
cond_keys = ['where', 'on', 'detects']


def owners(spec):
    """ The variable of each declared disease in the code (see `disease_code()`), with the names in it that a condition can use: its states, transitions, and stratifications """
    out = {}
    for mod in [mod for mod in spec['modules'] if 'states' in mod]: # The declared ones
        var = mod['name'] if mod['name'].isidentifier() else 'disease'
        out[mod['name']] = (var, {s['name'] for s in mod['states']} | {t['name'] for t in mod['transitions']} | {g['name'] for g in mod['stratifications']})
    return out


def cond_code(text, mods):
    """ A condition written as text in the model spec (e.g. "(people.age < 5)") as Python: an attribute of people from ss.ppl, and a state, transition, or stratification from its disease """
    class Names(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'people':
                return ast.Attribute(ast.Attribute(ast.Name('ss'), 'ppl'), node.attr)
            elif isinstance(node.value, ast.Name) and node.value.id in mods:
                return ast.Attribute(ast.Name(mods[node.value.id][0]), node.attr)
            return self.generic_visit(node)
        def visit_Name(self, node):
            if node.id in mods: # A delivery or product, as an event
                return ast.Name(mods[node.id][0])
            having = [var for var,names in mods.values() if node.id in names]
            if having:
                return ast.Attribute(ast.Name(having[0]), node.id)
            return ast.parse(f"ss.conditions.Cond('dim', {node.id!r})", mode='eval').body # A stratification of the sim
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id not in mods and hasattr(ss, node.func.id): # A typed value, e.g. years(5)
                node.func = ast.Attribute(ast.Name('ss'), node.func.id)
                node.args = [self.visit(arg) for arg in node.args]
                return node
            return self.generic_visit(node)
    return ast.unparse(Names().visit(ast.parse(text, mode='eval').body))


def product_code(prod, mods):
    """ A product of a delivery in the model spec as Python """
    kwargs = product_kwargs(prod, mods)
    kwargs.append(f'name={prod["name"]!r}')
    return f"ss.{prod['type']}({', '.join(kwargs)})"


def product_kwargs(prod, mods):
    """ The keywords that make a product, as Python """
    return [f'{k}={cond_code(v, mods) if k in cond_keys and isinstance(v, str) and not isinstance(v, Code) else code_value(v)}' for k,v in prod.items() if k not in ['type', 'name', 'made']]


def delivery_code(iv, mods):
    """ A delivery in the model spec as Python, with its product as the variable that holds it (see `interventions_code()`) """
    def value(key, val):
        if key in cond_keys and isinstance(val, str) and not isinstance(val, Code):
            return cond_code(val, mods)
        return code_value(val)
    args = []
    prod = iv.get('product_spec')
    inline = []
    if prod is not None and prod['name'] not in mods: # Made by the delivery from its own keywords, e.g. ss.vaccinate(efficacy=0.9, ...): see interventions_code()
        inline = product_kwargs(prod, mods)
    elif prod is not None:
        args.append(mods[prod['name']][0])
    if isinstance(iv.get('move'), str): # Between states, written with the disease's states
        dis = mods.get(iv.get('disease'), [None])[0]
        src, dst = iv['move'].split(' >> ')
        state = lambda name: f'{dis}.{name}' if dis else f'ss.State({name!r})'
        args.append(f"move={' + '.join(state(name) for name in src.split(' + '))} >> {state(dst)}")
    elif iv.get('move') is not None:
        args.append(move_code(iv['move'], f"{mods.get(iv.get('disease'), ['disease'])[0]}.strains"))
    if iv.get('increment'):
        args.append(f'increment={increment_code(iv["increment"])}')
    args += inline + [f'{k}={value(k, iv[k])}' for k in delivery_keys if iv.get(k) is not None]
    args.append(f'name={iv["name"]!r}')
    return f"ss.{iv['type']}({', '.join(args)})"


def interventions_code(mods, diseases, lines):
    """
    The deliveries and modifiers of the model spec as Python: each product and delivery as a variable (so that a step can be
    triggered by another's event, e.g. `who=screening.positive`), defined before the steps that refer to them; returns what goes in
    `interventions=`, and the lines that define them are added to `lines`
    """
    taken = {var for var,_ in diseases.values()} | {'ss', 'np', 'sim'}
    names = {}
    def var(name):
        out = name if name.isidentifier() and not keyword.iskeyword(name) else 'step'
        while out in taken:
            out = f'{out}_'
        taken.add(out)
        return out
    deliveries = [m for m in mods if m.get('type') in delivery_types]
    used = lambda iv: {n.id for k in cond_keys for v in sc.tolist(iv.get(k)) if isinstance(v, str) and not isinstance(v, Code) for n in ast.walk(ast.parse(v, mode='eval')) if isinstance(n, ast.Name)}
    held = [iv['product_spec']['name'] for iv in deliveries if iv.get('product_spec') is not None]
    named = set().union(*[used(iv) for iv in deliveries])
    prods = {iv['product_spec']['name']:iv['product_spec'] for iv in deliveries if iv.get('product_spec') is not None}
    prods = {name:prod for name,prod in prods.items() if not (prod.get('made') and held.count(name) == 1 and name not in named)} # A product its delivery made, used nowhere else, is written in the delivery's call
    for prod in prods.values():
        names[prod['name']] = (var(prod['name']), set())
    for m in deliveries:
        names[m['name']] = (var(m['name']), set())
    refs = sc.mergedicts(diseases, names)
    defined = set()
    for prod in prods.values():
        lines.append(f"{names[prod['name']][0]} = {product_code(prod, refs)}")
        defined.add(prod['name'])
    todo = list(deliveries) # Each after the deliveries its conditions name, where it can be
    while todo:
        ready = [iv for iv in todo if not ({n for n in used(iv) if n in names} - defined - {iv['name']})] or todo[:1]
        iv = ready[0]
        lines.append(f"{names[iv['name']][0]} = {delivery_code(iv, refs)}")
        defined.add(iv['name'])
        todo.remove(iv)
    out = []
    for m in mods: # In the order given (anything else is listed as not written: see to_code())
        if m.get('type') == 'modify':
            out.append(modify_code(m, diseases))
        elif contacts_code(m):
            out.append(contacts_code(m))
        elif m.get('type') in delivery_types:
            out.append(names[m['name']][0])
    return out


def to_code(spec, filename=None, header=None, footer=None, code=None, missing=None):
    """
    Write a model spec as a Python file that builds the same sim

    Only the declared parts of a model are written: the declared diseases, in the operator
    form (`S >> I @ beta * I`), with each parameter and state declared first, and each
    function used in the model defined before them (from its source); contact matrices;
    `ss.modify`; and the deliveries. Everything else is not reconstructed: modules and
    networks written as code (named by the class that makes them, `factory`, with their
    parameters in the spec), other interventions, movement between places from data,
    likelihoods other than a distribution, and functions whose source is not a plain
    definition are listed in comments at the end, and in a warning (`ss.errors.LossyExport`),
    since the code then builds a different model until they are added by hand. A part of
    a declared disease that cannot be written (the migration of a transition, custom
    functions of the disease) raises `NotImplementedError` rather than write a different model.

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
    prelude = [] # The lines that define the deliveries, after the diseases
    for key,simkey in [('networks', 'networks'), ('modifiers', 'connectors'), ('interventions', 'interventions')]:
        if key in code or simkey in code:
            extra[simkey] = code.get(simkey, code.get(key))
            continue
        mods = with_functions(spec.get(key) or [], funcs, left)
        for mod in mods:
            if mod.get('type') not in ['modify'] + delivery_types and not contacts_code(mod):
                left.append(f'{key[:-1]} "{mod.get("name", "?")}" ({mod.get("type", "?")}): add it to ss.Sim({simkey}=...)')
        written = interventions_code(mods, owners(spec), prelude) # Deliveries as variables, defined before the sim
        if written:
            extra[simkey] = '[' + ', '.join(written) + ']'
    lines = []
    if header:
        lines += ['"""', *header.strip().splitlines(), '"""']
    lines += ['import starsim as ss', '']
    if funcs:
        lines += ['# Functions'] + [line for block in funcs for line in block.splitlines() + ['']]
    names = []
    families = [mod for mod in spec['modules'] if mod['type'] == 'Family']
    copies = {fam['members'][0]:fam for fam in families if fam['strains']['members'] is None} # Strains with courses of their own, copied from one natural history: written as it, once
    skip = {name for fam in copies.values() for name in fam['members'][1:]}
    for mod in [mod for mod in spec['modules'] if mod['type'] != 'Family' and mod['name'] not in skip]:
        if mod['name'] in copies:
            fam = copies[mod['name']]
            names.append(disease_code(dict(mod, name=fam['name'], init_prev=fam['init_prev'], own_strains=fam['strains']), lines, left))
        elif mod['type'] == 'Disease':
            names.append(disease_code(mod, lines, left))
        else: # Written as code (or declared, but not a disease): by the class that makes it, with its parameters, which the user can add
            left.append(f'module "{mod["name"]}" ({mod.get("factory") or mod["type"]}): make it and add it to ss.Sim(...)')
    for fam in [fam for fam in families if fam['members'][0] not in copies]: # Strains with courses of their own after their strains, in their place among the sim's diseases
        names = strains_code(fam, names, lines, left)
    if prelude:
        lines += ['# Interventions'] + prelude + ['']
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
        ss.warn(f'the code builds a different model until these are added by hand: {"; ".join(left)}', ss.errors.LossyExport)
    if footer:
        lines += [''] + [f'# {line}' for line in footer]
    lines += ['', "if __name__ == '__main__':", '    sim.run()', '    sim.plot()', '']
    code = '\n'.join(lines)
    if any('np.' in line for line in lines if not line.startswith('#')):
        code = code.replace('import starsim as ss\n', 'import numpy as np\nimport starsim as ss\n', 1)
    if any('ssl.' in line for line in lines if not line.startswith('#')): # A module from the library, e.g. ssl.DegreeNet from an import
        code = code.replace('import starsim as ss\n', 'import starsim as ss\nimport starsim.library as ssl\n', 1)
    if any('expit(' in line for line in lines if not line.startswith('#')):
        code = code.replace('import starsim as ss\n', 'import starsim as ss\nfrom scipy.special import expit\n', 1)
    if filename is not None:
        sc.savetext(filename, code)
    return code


#%% The spec of a sim: to_spec(), hash(), and to_code()

def built_copy(sim):
    """ A built copy of a sim, built quietly (without the lines a run prints as it starts), for what reads the built model, e.g. `hash()` and `to_spec()` """
    sim = sc.dcp(sim)
    sim.pars.verbose = 0
    return sim.init()


def sim_hash(sim):
    """ The model's fingerprint: see `ss.Sim.hash()` """
    if not sim.initialized: # A copy, so that this sim is not changed
        return built_copy(sim).hash()
    spec = sim.to_spec()
    digest = ss.utils.short_hash
    model = dict(modules=spec['modules'], networks=spec['networks'], stratify=spec['stratify'], modifiers=spec['modifiers'])
    for mod in model['modules']: # Where a value came from is not part of the model
        if isinstance(mod.get('pars'), list): # A declared module's parameters
            for p in mod['pars']:
                p.pop('source', None)
        if isinstance(mod.get('transitions'), list): # The order of the transitions is not part of the model
            mod['transitions'] = sorted(mod['transitions'], key=lambda t: t['name'])
        dis = sim.diseases.get(mod.get('name')) if mod.get('init_prev') else None
        if getattr(dis, 'graph', None) is not None and dis.graph.transmissions and len(dis.graph.transmissions[0].dst) == 1: # init_prev=0.01 is init=dict(I=0.01), the same model
            state = next(s for s in mod['states'] if s['name'] == dis.graph.transmissions[0].dst[0].name)
            if state.get('init') is None:
                state['init'], mod['init_prev'] = mod['init_prev'], None
    unknown = ss.presentation.unknown_funcs(dict(model, interventions=spec['interventions']))
    if unknown:
        errormsg = f'sim.hash() cannot tell these functions apart from others, since their source is not available and they have no name to import them by: {sc.strjoin(unknown)}. Define each with def, or give it a version (e.g. fn.version = "1")'
        raise ValueError(errormsg)
    structure, values = ss.utils.split_numbers(model, unit=sim.unit, skip=['guessed', 'label', 'doc'])
    config = {k:v for k,v in spec['config'].items() if k != 'rand_seed'}
    data = None if sim.data is None else digest(sim.data.to_csv())
    return sc.objdict(model=digest(structure), pars=digest(values), config=digest(config), scenario=digest(spec['interventions']), data=data, seed=sim.pars.rand_seed)


def sim_spec(sim, filename=None, **kwargs):
    """ The model spec of a sim: see `ss.Sim.to_spec()` """
    if not sim.initialized: # A copy, so that this sim is not changed
        return built_copy(sim).to_spec(filename=filename, **kwargs)
    pars = sim.pars
    def tstr(val):
        if isinstance(val, ss.dur) and type(val).base != sim.unit_base + 's': # A duration in another unit than the sim's (e.g. start=2000 in years, with rates per day) is written with its unit
            return ss.transitions.quantity_json(val, 'dur', None)
        elif isinstance(val, ss.dur):
            return float(val.value)
        return val.strftime('%Y-%m-%d') if isinstance(val, ss.date) and val == val.normalize() else str(val) # Dates as ISO 8601
    out = sc.objdict()
    out.schema = 'lingua-franca/1.0'
    out.modules = [mod.to_spec() for key in ['custom', 'demographics', 'diseases'] for mod in sim[key]() if not isinstance(mod, ss.demographics.Flows)] # Births and deaths declared for the population are, by now, transitions of a disease
    out.networks = [mod.to_spec() for mod in list(sim.networks()) + getattr(sim, 'mixed', [])]
    out.stratify = [sc.dcp(strat.spec) for strat in sim.groups.sim_strats] # Those of a module are in its own spec, and those of an intervention are part of the scenario
    out.modifiers = [mod.to_spec() for mod in sim.connectors()]
    out.interventions = [mod.to_spec() for mod in sim.interventions()]
    out.config = sc.objdict(solver=pars.solver, integrator=pars.integrator, approx=pars.approx, unit=sim.unit or pars.unit, # A sim of no declared modules has only the unit it was given
                            start=tstr(sim.t.start), stop=tstr(sim.t.stop), dt=float(sim.t.dt.to(sim.unit_base).value),
                            total_pop=float(pars.total_pop), n_agents=int(pars.n_agents) if sim.solver == 'abm' else None, rand_seed=pars.rand_seed)
    out.config.crn = ss.options.crn # Numerical options that change the results
    out.config.precision = ss.options.precision
    out.config.substeps = pars.substeps
    out.config.use_aging = bool(pars.use_aging) if 'use_aging' in sim.given else None # Only when given, e.g. an age stratification without aging
    if getattr(sim, 'named_people', None) is not None and sim.people_from is None: # Where a named population came from (people that come with a contact matrix come with it again)
        out.config.people = dict(sim.named_people.ref, total=sim.named_people.total)
    out.meta = sc.objdict(library_version=ss.__version__, solver_chosen=sim.solver, solver_reason=sim.solver_reason, notes=list(sim.notes))
    out = sc.jsonify(out)
    if filename is not None:
        return sc.savejson(filename, out, **kwargs)
    return out


def sim_code(sim, filename=None):
    """ The sim as a Python file that builds it again: see `ss.Sim.to_code()` """
    header, footer, code = None, None, None
    missing = [] if sim.data is None else [f'the data ({len(sim.data.columns)} series: {sc.strjoin(sim.data.columns)}): pass it with ss.Sim(..., data=...)']
    imp = getattr(sim, 'imported', None)
    if imp is not None:
        code = imp.get('code')
        source = sc.path(imp.file).name if imp.file else 'an object'
        header = f'Imported from {imp.format} ({source}) by ss.interop.load()\n\nHow each rate was read:\n' + '\n'.join(imp.choices)
        if imp.incomplete:
            footer = ['Not translated (ImportIncomplete):'] + [f'  {name}: {why}' for name,why in imp.incomplete]
    return ss.to_code(sim.to_spec(), filename=filename, header=header, footer=footer, code=code, missing=missing)


#%% Comparing two models

def shown(val, unit=None):
    """ A value from the model spec, short and in the sim's unit where it has one, e.g. "0.5/day" or "lognorm_ex(mean=10 days, std=3 days)" """
    if isinstance(val, dict) and val.get('type') in ['dur', 'per', 'prob', 'freq'] and val.get('unit'):
        num, u = val['value'], val['unit']
        if unit and u != unit:
            factor = ss.time.get_dur_class(u)(1).to(unit).value
            num, u = (num*factor if val['type'] == 'dur' else num/factor), unit
        num = f'{num:.6g}'
        return f'{num} {u}{"" if num == "1" else "s"}' if val['type'] == 'dur' else f'{num}/{u}' if val['type'] == 'per' else f'{val["type"]} {num}/{u}'
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
        if 'Family' in (x['type'], y['type']): # Strains with courses of their own: their strains and options
            for key in ['members', 'shared', 'immunity', 'init_prev', 'split', 'values']:
                vx, vy = (x if key in ['members', 'init_prev'] else x.get('strains', {})).get(key), (y if key in ['members', 'init_prev'] else y.get('strains', {})).get(key)
                if key == 'immunity' and vx and vy and len(vx) == len(vy): # Each, by what it protects with and what it lowers (the structure), then by its values
                    for i,(ix,iy) in enumerate(zip(vx, vy)):
                        for k in sorted(set(ix) | set(iy)):
                            if ix.get(k) != iy.get(k):
                                (pars if k in ['rel_sus', 'rel_trans', 'dur', 'p'] and k in ix and k in iy else model).append(f'{prefix}immunity[{i}].{k} {shown(ix.get(k))} -> {shown(iy.get(k))}')
                elif vx != vy:
                    (pars if key in ['init_prev', 'split', 'values'] and None not in (vx, vy) else model).append(f'{prefix}{key} {shown(vx, unit)} -> {shown(vy, unit)}')
            continue
        as_dict = lambda pars: dict(pars) if isinstance(pars, dict) else {p['name']:p['value'] for p in pars} # A module written as code keeps its parameters by name
        px, py = as_dict(x.get('pars') or []), as_dict(y.get('pars') or [])

        # States
        sx, sy = [s['name'] for s in x.get('states', [])], [s['name'] for s in y.get('states', [])]
        model += [f"{prefix}state '{s}' removed" for s in sx if s not in sy] + [f"{prefix}state '{s}' added" for s in sy if s not in sx]
        first = x.get('first', sx[0] if sx else None)
        if x.get('first') != y.get('first') and None not in (x.get('first'), y.get('first')):
            model.append(f"{prefix}first {x['first']} -> {y['first']}")
        for s in [s for s in x.get('states', []) if s['name'] in sy]:
            t = next(o for o in y['states'] if o['name'] == s['name'])
            for key in ['people', 'rel_sus', 'rel_trans', 'init', 'init_n']:
                rest = key in ['init', 'init_n'] and s['name'] == first and None in (s.get(key), t.get(key)) # The first state starts with the rest, whether or not it is written
                if s.get(key) != t.get(key) and not rest:
                    (model if key == 'people' else pars).append(f"{prefix}{s['name']}.{key} {shown(s.get(key))} -> {shown(t.get(key))}")

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
