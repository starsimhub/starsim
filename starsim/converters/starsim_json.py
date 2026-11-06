"""
Starsim's own form of a model as JSON: `ss.export_to(sim, 'json')` writes it, and `ss.import_from('model.json')` reads it

The file is a declared model as plain data, in the words of its code: each value and
rate is written as it would be in Python without the `ss.` (`"days(10)"`,
`"S >> I @ beta * I"`), each stratification is the arguments of `stratify()`, and
what was left at its default is left out. For example:

    {
      "starsim": "4.0.0",
      "stop": "days(200)",
      "diseases": [{
        "name": "sir",
        "pars": {"beta": "perday(0.3)", "dur_inf": {"value": "days(10)", "bounds": ["days(5)", "days(20)"]}},
        "states": ["S", "I", "R"],
        "transitions": {
          "infection": "S >> I @ beta * I",
          "recovery":  "I >> R @ dur_inf"
        },
        "init": {"I": 0.01},
        "stratify": [{"patch": ["urban", "rural"]}]
      }],
      "networks": [{"type": "RandomNet", "n_contacts": 5}]
    }

It is one more format, not a second form of the model: the model is its code, and an
import writes that code (`sim.imported.code`) and runs it, as for every other format.
Each expression is read with a parser that allows only the model's own names and
Starsim's and NumPy's functions, so reading a file runs no code from it.

What it carries: the sim's settings; each declared disease's parameters (with their
bounds, priors, and descriptions), states, transitions (rates, dwell times, splits, and
minimum stays), initial values, results, and stratifications; and every other module
(networks, interventions, ...) by its type and the parameters that differ from its
defaults, as Starsim 3's dict input gave them. What it does not carry yet (outcomes,
modifiers, functions, strains, transfers, conditions, a module of the user's own class)
is listed as `ss.errors.LossyExport`.

Used with: agents and compartments; declared modules
"""
import ast
import json
import keyword
import numpy as np
import sciris as sc
import starsim as ss
from .core import code_value, time_names, func_names

__all__ = ['to_json', 'load_json']

# The settings of a sim that the file carries, as they were given
settings = ['label', 'n_agents', 'total_pop', 'pop_scale', 'start', 'stop', 'dur', 'dt', 'unit', 'solver', 'rand_seed', 'branching', 'approximations', 'integrator', 'substeps', 'aging']
module_keys = ['networks', 'demographics', 'interventions', 'analyzers', 'connectors']
values = ['par', 'seasonal', 'peract', 'timeseries', 'by', 'Rate', 'date', 'datedur', 'dur'] # Starsim's values that can be written as calls, besides typed quantities and distributions
dist_names = sorted(name for name in dir(ss) if isinstance(getattr(ss, name), type) and issubclass(getattr(ss, name), ss.Dist) and name != 'Dist')
constants = dict(pi='np.pi', inf='np.inf', nan='np.nan', null='ss.null', stay='ss.stay')


#%% Writing

def drop_prefixes(code):
    """ Python as the file writes it: `ss.days(10)` as `days(10)`, `np.exp(x)` as `exp(x)`, and `ss.person.age` as `people.age` """
    class Drop(ast.NodeTransformer):
        def visit_Attribute(self, n):
            if isinstance(n.value, ast.Attribute) and isinstance(n.value.value, ast.Name) and n.value.value.id == 'ss' and n.value.attr == 'person':
                return ast.Attribute(ast.Name('people'), n.attr)
            elif isinstance(n.value, ast.Name) and n.value.id in ['ss', 'np']:
                return ast.Name(n.attr)
            self.generic_visit(n)
            return n
    return ast.unparse(Drop().visit(ast.parse(code, mode='eval')))


class Writer:
    """ The parts of a sim as plain data, and what was left out """
    def __init__(self):
        self.lost = []
        return

    def value(self, val, pars=None):
        """ A value as JSON: numbers, text, lists, and dicts as they are, `ss.by` as `{"by": {...}}`, a parameter as its name, and anything else as its code without the `ss.` """
        pars = pars or {}
        v = lambda x: self.value(x, pars)
        if isinstance(val, ss.par):
            name = next((k for k,p in pars.items() if p is val), None)
            return name if name is not None else v(val.raw)
        elif isinstance(val, np.generic):
            return v(val.item())
        elif isinstance(val, float) and not np.isfinite(val):
            return 'nan' if np.isnan(val) else 'inf' if val > 0 else '-inf'
        elif val is None or isinstance(val, (bool, str)) or sc.isnumber(val):
            return val
        elif isinstance(val, (list, tuple, np.ndarray)):
            return [v(x) for x in val]
        elif isinstance(val, dict):
            return {str(k):v(x) for k,x in val.items()}
        elif isinstance(val, ss.by) and val.given is not None and val.array is None and val.table is None:
            return {'by': v(val.given)}
        elif isinstance(val, ss.expressions.Expr):
            return self.expr(val, pars)
        return drop_prefixes(code_value(val, pars=pars))

    def expr(self, val, pars):
        """ A rate or another expression, as its text; a bare parameter is its name """
        if isinstance(val, ss.par):
            return self.value(val, pars)
        elif sc.isnumber(val):
            return val
        return drop_prefixes(str(val))

    def par(self, p, pars):
        """ A parameter: its value, or its value with its bounds, prior, and description """
        out = sc.objdict(value=self.value(p.raw, pars))
        for key in ['bounds', 'prior', 'doc', 'label']:
            val = getattr(p, key, None)
            if val is not None:
                out[key] = self.value(val, pars)
        return out.value if len(out) == 1 else dict(out)

    def transition(self, tr, g):
        """ A transition: as one line in the words of its code (`"I >> R @ dur_inf"`) if it has only its ends and a rate or a dwell time, else as the arguments of `ss.Transition()` """
        pars = g.pars
        ends = lambda names, empty: empty if not names else names[0] if len(names) == 1 else '[' + ', '.join(names) + ']'
        name = lambda x: 'null' if x is ss.null else 'stay' if x is ss.stay else x.name
        src = ends([name(x) for x in tr.source], 'null')
        dst = [name(x) for x in tr.dest or []] + ['stay']*bool(tr.stays) + ['null']*bool(tr.dies) # ss.stay and ss.null are kept out of dst, each with its own p
        out = sc.objdict(source=src if src != 'null' else None, dest=None if not dst else dst[0] if len(dst) == 1 else dst)
        dur = tr.dur
        if tr.rate is not None:
            out.rate = self.expr(tr.rate, pars)
        elif dur is not None:
            out.dur = self.expr(dur, pars)
        p = tr.p
        if p is not None: # A split, or a share of the people leaving by a transition with one destination (e.g. recovery with p=1 - p_death beside death with p=p_death)
            p = sc.tolist(p) + [tr.p_stay]*bool(tr.stays) + [tr.p_die]*bool(tr.dies)
            out.p = [None if x is None else self.expr(x, pars) for x in p]
        for key in ['min_stay', 'label', 'stages']:
            val = getattr(tr, key, None)
            if val is not None:
                out[key] = self.expr(val, pars) if key == 'min_stay' else val
        for key in ['where', 'move', 'increment']:
            if getattr(tr, key, None) is not None:
                self.lost.append(f'not exported: the {key}= of transition "{g.module.name}.{tr.name}"')
        typed = isinstance(getattr(dur, 'raw', dur), (ss.Duration, ss.Dist)) # A dwell time written with @ needs a unit, so that it is not read as a rate
        if set(out.keys()) <= {'source', 'dest', 'rate', 'dur'} and not isinstance(out.dest, list) and ('rate' in out or typed):
            rate = out.get('rate', out.get('dur'))
            return f'{src} >> {out.dest or "null"} @ {rate}'
        return {k:v for k,v in out.items() if v is not None}

    def strat(self, s):
        """ A stratification, as the arguments of `stratify()` """
        if s.geo is not None or s.mobility is not None:
            self.lost.append(f'changed: the places of "{s.name}" are written as names, without their geography or movement')
        out = {s.name: self.value(s.edges if s.edges is not None else s.labels)}
        if s.fractions_given:
            out['fractions'] = self.value(s.fractions)
        if s.adjust:
            out['adjust'] = self.value(dict(s.adjust))
        default = 'age' if s.name == 'age' else 'plain'
        if s.kind != default:
            out['kind'] = s.kind
        for key in ['states', 'where']:
            if getattr(s, key) is not None:
                out[key] = self.value(getattr(s, key))
        return out

    def disease(self, mod):
        """ A declared disease """
        g = mod.graph
        pars = g.pars
        out = sc.objdict(name=mod.name)
        if g.pars:
            out.pars = {name:self.par(p, pars) for name,p in pars.items() if p is not g.init_prev}
        out.states = []
        for name,s in g.states.items():
            kw = {k:self.value(v, pars) for k,v in dict(people=s.people if not s.people else None, label=s.label, rel_sus=s.rel_sus, rel_trans=s.rel_trans, keep_time=s.keep_time or None).items()
                  if v is not None and not (k.startswith('rel_') and sc.isnumber(v) and v == 1)}
            out.states.append(dict(name=name, **kw) if kw else name)
        out.transitions = {name:self.transition(tr, g) for name,tr in g.transitions.items()}
        results = {name:self.expr(e, pars) for name,e in g.exprs.items() if not (name == 'prevalence' and g.auto_prevalence)}
        if results: # A result used in a rate (or in another result) is written there by its name, as the code would
            fold = Folder(results)
            out.transitions = {name:fold.transition(tr) for name,tr in out.transitions.items()}
            out.results = {name:fold(text, skip=name) for name,text in results.items()}
        else:
            fold = lambda text: text
        for key in ['init', 'init_n']:
            vals = {**{k:getattr(st, key) for k,st in g.states.items() if getattr(st, key) is not None}, **getattr(g, key)}
            bad = [k for k,v in vals.items() if callable(v) and not isinstance(v, ss.expressions.Expr)]
            for k in bad:
                self.lost.append(f'not exported: the initial value of "{mod.name}.{k}", a function')
            vals = {k:fold(self.value(v, pars)) for k,v in vals.items() if k not in bad}
            if vals:
                out[key] = vals
        if g.init_prev is not None and 'init_prev' not in g.guessed: # Not the default, which is worked out again
            out.init_prev = self.value(getattr(g.init_prev, 'raw', g.init_prev))
        if g.default_given is not None:
            out.default_state = g.default_given
        strats = [self.strat(s) for s in g.strats if s.kind != 'strain']
        if strats:
            out.stratify = strats
        for key,what in [('outcomes', 'outcome'), ('modifiers', 'modifier'), ('funcs', 'function'), ('attrs', 'attribute'), ('transfers', 'transfer')]:
            for name in getattr(g, key):
                self.lost.append(f'not exported: {what} "{mod.name}.{name}"')
        if g.derived:
            self.lost.append(f'not exported: the stratification of "{mod.name}" by {sc.strjoin(s.name for s in g.derived)}, a derived attribute')
        if getattr(mod, 'strains', None) is not None:
            self.lost.append(f'not exported: the strains of "{mod.name}"')
        return dict(out)

    def module(self, mod):
        """ Another module, by its type and the parameters that differ from its defaults, or None if it is the user's own class """
        cls = type(mod)
        name = cls.__name__
        if getattr(ss, name, None) is cls:
            typename = name
        elif getattr(ss.library, name, None) is cls if hasattr(ss, 'library') else False:
            typename = f'ssl.{name}'
        else:
            self.lost.append(f'not exported: module "{mod.name}" ({name}), which is not one of Starsim\'s')
            return None
        out = dict(type=typename)
        try:
            default = cls()
        except Exception: # A module that needs arguments: every parameter is written
            default = None
        for key,val in mod.pars.items():
            mine = self.value(val)
            if default is None or key not in default.pars or self.value(default.pars[key]) != mine:
                out[key] = mine
        if default is not None and mod.name != default.name:
            out['name'] = mod.name
        return out


class Folder:
    """ Write each part of an expression that is one of the results as the result's name, the largest first """
    def __init__(self, results):
        self.forms = []
        for name,text in results.items():
            node = ast.parse(text, mode='eval').body
            if not isinstance(node, (ast.Name, ast.Constant)):
                self.forms.append((name, ast.dump(node), len(text)))
        self.forms.sort(key=lambda f: -f[2])
        return

    def __call__(self, text, skip=None):
        if not isinstance(text, str) or not self.forms:
            return text
        forms = {dump:name for name,dump,size in self.forms if name != skip and (skip is None or size < len(text))}
        class Fold(ast.NodeTransformer):
            def visit(self, n):
                name = forms.get(ast.dump(n)) if isinstance(n, ast.expr) else None
                return ast.Name(name) if name else super().visit(n)
        try:
            return ast.unparse(Fold().visit(ast.parse(text, mode='eval').body))
        except SyntaxError:
            return text

    def transition(self, tr):
        """ A transition, in its one line or as its arguments """
        if isinstance(tr, str):
            ends, rate = tr.split(' @ ', 1)
            return f'{ends} @ {self(rate)}'
        return {k:[self(x) for x in v] if k == 'p' else self(v) if k in ['rate', 'dur'] else v for k,v in tr.items()}


def to_json(sim):
    """
    A sim as Starsim's own JSON form (see the module's docstring), and what it could not carry

    Args:
        sim (`ss.Sim`): the sim
    """
    base = sim.__dict__.get('_before_init', None) if sim.initialized else sim # The settings as they were given
    base = base if isinstance(base, ss.Sim) else sim
    sim = ss.converters.export.initialized(sim)
    w = Writer()
    out = sc.objdict(starsim=ss.__version__)
    defaults = ss.SimPars()
    for key in settings:
        val = w.value(base.pars.get(key))
        if val is not None and val != w.value(defaults.get(key)):
            out[key] = val
    if base.pars.get('stratify') is not None:
        w.lost.append('not exported: the sim\'s stratification (sim.stratify())')
    if base.pars.get('people') is not None:
        w.lost.append('not exported: the people (people=)')
    diseases = [w.disease(mod) if getattr(mod, 'graph', None) is not None else w.module(mod) for mod in sim.diseases.values()] # A disease written as code is one of Starsim's, by its type
    diseases = [d for d in diseases if d is not None]
    if diseases:
        out.diseases = diseases
    for key in module_keys:
        skip = lambda m: isinstance(m, (ss.demographics.Flows, ss.modify)) # Births and deaths declared for the population are transitions of a disease by now, and a disease's modifiers are listed with it
        mods = [w.module(m) for m in sim[key]() if not skip(m)]
        for m in sim[key]():
            if isinstance(m, ss.modify) and not any(m is x for d in sim.diseases.values() for x in getattr(getattr(d, 'graph', None), 'modifiers', {}).values()):
                w.lost.append(f'not exported: modifier "{m.name}"')
        mods = [m for m in mods if m is not None]
        if mods:
            out[key] = mods
    text = dumps(dict(out)) + '\n'
    count = lambda n, word: f'{n} {word if n == 1 else word + "s"}'
    summary = f'Exported to Starsim\'s JSON form: {count(len(diseases), "disease")}' + ''.join(f', {count(len(out[k]), k.rstrip("s"))}' for k in module_keys if k in out)
    return text, summary, w.lost


def dumps(obj, indent=0, width=100):
    """ JSON with each list and dict on one line if it fits, so the file reads like the code """
    flat = json.dumps(obj)
    if not isinstance(obj, (list, dict)) or len(flat) + indent <= width:
        return flat
    pad = ' '*(indent + 2)
    if isinstance(obj, list):
        items = [pad + dumps(v, indent + 2, width) for v in obj]
        return '[\n' + ',\n'.join(items) + '\n' + ' '*indent + ']'
    items = [f'{pad}{json.dumps(k)}: {dumps(v, indent + 2, width)}' for k,v in obj.items()]
    return '{\n' + ',\n'.join(items) + '\n' + ' '*indent + '}'


#%% Reading

class Reader:
    """ Starsim's JSON form as Starsim code, checking every name in it """
    def __init__(self, file=None):
        self.file = file
        return

    def error(self, where, msg):
        raise ValueError(f'In {self.file or "the model"}, {where}: {msg}')

    def expr(self, text, names, where):
        """ An expression as Python: names of the model, Starsim's typed quantities, distributions, and values (with `ss.`), and NumPy's functions (with `np.`) """
        calls = set(time_names) | set(dist_names) | set(values) | set(func_names) | {'dict', 'mean'}
        def visit(n):
            if isinstance(n, ast.Constant) and (n.value is None or isinstance(n.value, (bool, str)) or sc.isnumber(n.value)):
                return n
            elif isinstance(n, ast.Name):
                if n.id in names:
                    return n
                elif n.id in constants:
                    return ast.parse(constants[n.id], mode='eval').body
                self.error(where, f'"{n.id}" is not a state, parameter, or transition of the model, in "{text}"')
            elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'people':
                return ast.Attribute(ast.Attribute(ast.Name('ss'), 'person'), n.attr)
            elif isinstance(n, ast.Attribute) and n.attr in ['n', 'frac', 'total', 'dwell', 'time_in']:
                return ast.Attribute(visit(n.value), n.attr)
            elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.BitAnd, ast.BitOr)):
                return ast.BinOp(visit(n.left), n.op, visit(n.right))
            elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd, ast.Invert)):
                return ast.UnaryOp(n.op, visit(n.operand))
            elif isinstance(n, ast.Compare) and all(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)) for op in n.ops):
                return ast.Compare(visit(n.left), n.ops, [visit(c) for c in n.comparators])
            elif isinstance(n, (ast.List, ast.Tuple)):
                return type(n)([visit(e) for e in n.elts], ast.Load())
            elif isinstance(n, ast.Dict):
                return ast.Dict([visit(k) for k in n.keys], [visit(v) for v in n.values])
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in calls:
                name = n.func.id
                func = ast.Name(name) if name in ['dict', 'expit'] else ast.Attribute(ast.Name('np' if name in func_names else 'ss'), name)
                if name == 'mean':
                    func = ast.Attribute(ast.Name('ss'), 'mean')
                elif name == 'dur': # The class of a duration, ss.Duration
                    func = ast.Attribute(ast.Name('ss'), 'Duration')
                return ast.Call(func, [visit(a) for a in n.args], [ast.keyword(k.arg, visit(k.value)) for k in n.keywords if k.arg is not None])
            self.error(where, f'"{ast.unparse(n)}" is not something a model can say, in "{text}"')
        try:
            node = ast.parse(str(text).strip(), mode='eval').body
        except SyntaxError:
            self.error(where, f'"{text}" is not an expression')
        return ast.unparse(visit(node))

    def is_expr(self, text, names):
        """ Whether text is an expression (a typed value, a call, a name of the model) rather than plain text, e.g. a date or the name of a network """
        try:
            node = ast.parse(text.strip(), mode='eval').body
        except SyntaxError:
            return False
        known = set(names) | set(constants)
        funcs = {id(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)}
        return all(n.id in known or id(n) in funcs for n in ast.walk(node) if isinstance(n, ast.Name))

    def value(self, val, names, where):
        """ A value as Python: JSON's own values as they are, `{"by": {...}}` as `ss.by(...)`, and text that is an expression as its code """
        v = lambda x: self.value(x, names, where)
        if isinstance(val, str):
            return self.expr(val, names, where) if self.is_expr(val, names) else repr(val)
        elif isinstance(val, list):
            return '[' + ', '.join(v(x) for x in val) + ']'
        elif isinstance(val, dict):
            if list(val) == ['by']:
                return f'ss.by({kwargs(val["by"], v)})'
            if all(k.isidentifier() and not keyword.iskeyword(k) for k in val):
                return f'dict({kwargs(val, v)})'
            return '{' + ', '.join(f'{k!r}: {v(x)}' for k,x in val.items()) + '}'
        return repr(val)

    def ends(self, text, names, where):
        """ The source or destination of a transition: a state, `null`, or a list of states """
        code = self.expr(text, names, where)
        node = ast.parse(code, mode='eval').body
        ok = lambda n: isinstance(n, ast.Name) or ast.unparse(n) in ['ss.null', 'ss.stay']
        if not (ok(node) or isinstance(node, ast.List) and all(ok(e) for e in node.elts)):
            self.error(where, f'"{text}" is not a state, null, or a list of states')
        return code

    def transition(self, name, tr, names, where):
        """ The code of a transition, from its one line or its arguments """
        if isinstance(tr, str):
            if '>>' not in tr or '@' not in tr:
                self.error(where, f'"{tr}" is not a transition; write it as "S >> I @ rate", or as a dict of the arguments of ss.Transition()')
            ends, rate = [x.strip() for x in tr.split('@', 1)]
            src, dst = [x.strip() for x in ends.split('>>', 1)]
            rate = self.expr(rate, names, where)
            node = ast.parse(rate, mode='eval').body
            simple = isinstance(node, (ast.Name, ast.Constant, ast.Call, ast.Attribute))
            src = self.ends(src, names, where)
            if src.startswith('['): # People leaving any of several states, which the operators write as their sum
                src = '(' + ' + '.join(ast.unparse(e) for e in ast.parse(src, mode='eval').body.elts) + ')'
            return f'{src} >> {self.ends(dst, names, where)} @ {rate if simple else f"({rate})"}'
        unknown = set(tr) - {'source', 'dest', 'rate', 'dur', 'p', 'min_stay', 'label', 'stages'}
        if unknown:
            self.error(where, f'{sc.strjoin(sorted(unknown))} is not an argument of a transition here')
        args = [self.ends(tr.get('source') or 'null', names, where), self.ends(tr['dest'] if isinstance(tr.get('dest'), str) else '[' + ', '.join(x or 'null' for x in tr['dest']) + ']' if tr.get('dest') else 'null', names, where)]
        for key in ['rate', 'dur', 'p', 'min_stay', 'stages', 'label']:
            if tr.get(key) is not None:
                val = repr(tr[key]) if key == 'label' else self.value(tr[key], names, where)
                args.append(f'{key}={val}')
        return f'ss.Transition({", ".join(args)})'

    def disease(self, d, var):
        """ The lines of code of a declared disease, as the variable var """
        name = d.get('name', var)
        where = f'disease "{name}"'
        unknown = set(d) - {'name', 'label', 'pars', 'states', 'transitions', 'results', 'init', 'init_n', 'init_prev', 'default_state', 'stratify'}
        if unknown:
            self.error(where, f'{sc.strjoin(sorted(unknown))} is not part of a disease; its parts are name, pars, states, transitions, results, init, init_n, init_prev, default_state, and stratify')
        states = [s if isinstance(s, str) else s['name'] for s in d.get('states', [])]
        pars = list(d.get('pars', {}))
        transitions = list(d.get('transitions', {}))
        names = set(states) | set(pars) | set(transitions) | {'people'}
        lines = [f'# The disease "{name}"']

        # Parameters, each of which may use those before it
        for i,(k,p) in enumerate(d.get('pars', {}).items()):
            check_name(k, where, self)
            known = set(pars[:i])
            meta = p if isinstance(p, dict) and 'value' in p else dict(value=p)
            extra = ''.join(f', {key}={repr(meta[key]) if key in ["doc", "label"] else self.value(meta[key], known, where)}' for key in ['bounds', 'prior', 'doc', 'label'] if meta.get(key) is not None)
            lines.append(f'{k} = ss.par({self.value(meta["value"], known, f"{where}, parameter {k}")}{extra})')

        # States, with a run of plain ones declared together
        plain = []
        def flush():
            if len(plain) > 1:
                lines.append(f"{', '.join(plain)} = ss.states({' '.join(plain)!r})")
            elif plain:
                lines.append(f'{plain[0]} = ss.State({plain[0]!r})')
            plain.clear()
        for s in d.get('states', []):
            if isinstance(s, str):
                check_name(s, where, self)
                plain.append(s)
            else:
                check_name(s['name'], where, self)
                flush()
                unknown = set(s) - {'name', 'people', 'label', 'rel_sus', 'rel_trans', 'keep_time'}
                if unknown:
                    self.error(where, f'{sc.strjoin(sorted(unknown))} is not a property of a state; they are people, label, rel_sus, rel_trans, and keep_time')
                kw = ''.join(f', {k}={repr(s[k]) if k == "label" else self.value(s[k], set(pars), where)}' for k in ['people', 'label', 'rel_sus', 'rel_trans', 'keep_time'] if k in s)
                lines.append(f'{s["name"]} = ss.State({s["name"]!r}{kw})')
        flush()

        # Results, which rates may use, and transitions; a result of flows (e.g. incidence = infection / S) comes after them
        results = d.get('results', {})
        names |= set(results)
        before, after = [], []
        for k,e in results.items():
            check_name(k, where, self)
            text = self.expr(e, names, f'{where}, result {k}')
            uses = {n.id for n in ast.walk(ast.parse(text, mode='eval')) if isinstance(n, ast.Name)}
            (after if uses & set(transitions) else before).append(f'{k} = {text}')
        lines += before
        for k,tr in d.get('transitions', {}).items():
            check_name(k, where, self)
            lines.append(f'{k} = {self.transition(k, tr, names, f"{where}, transition {k}")}')
        lines += after
        args = pars + states + transitions
        kw = sc.objdict()
        for k in results:
            kw[k] = k
        for key in ['init', 'init_n']:
            if d.get(key):
                kw[key] = self.value(d[key], names, f'{where}, {key}')
        for key in ['init_prev']:
            if d.get(key) is not None:
                kw[key] = self.value(d[key], names, f'{where}, {key}')
        for key in ['default_state', 'label']:
            if d.get(key) is not None:
                kw[key] = repr(d[key])
        kw.name = repr(name)
        lines.append(f'{var} = ss.Disease(')
        for i in range(0, len(args), 8):
            lines.append(f'    {", ".join(args[i:i+8])},')
        for k,v in kw.items():
            lines.append(f'    {k} = {v},')
        lines.append(')')
        for strat in d.get('stratify', []):
            lines.append(f'{var}.stratify({kwargs(strat, lambda x: self.value(x, names, f"{where}, stratify"))})')
        return lines

    def module(self, m, key):
        """ The code of another module, from its type and parameters """
        m = dict(m)
        typename = m.pop('type', None)
        where = f'{key}, "{typename}"'
        lib, _, name = typename.rpartition('.') if typename else ('', '', '')
        space = ss if not lib else ss.library if lib == 'ssl' else None
        cls = getattr(space, name, None) if space is not None else None
        if not (isinstance(cls, type) and issubclass(cls, ss.Module)):
            self.error(where, f'"{typename}" is not a module of Starsim (e.g. "RandomNet") or of its library (e.g. "ssl.HouseholdNet")')
        return f'{"ssl" if lib else "ss"}.{name}({kwargs(m, lambda x: self.value(x, set(), where))})'


def check_name(name, where, reader):
    """ A name in the model must be a Python name, so that the code can use it """
    if not (isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)) or name in ['ss', 'ssl', 'np', 'sim', 'people', 'null', 'stay']:
        reader.error(where, f'"{name}" cannot be a name in a model: use letters, numbers, and underscores, not starting with a number, and not a word Python or Starsim uses')


def kwargs(d, v):
    """ Keyword arguments as code, with each value written by v """
    return ', '.join(f'{k}={v(x)}' for k,x in d.items())


def code(model, file=None):
    """ The Starsim code of a model in Starsim's JSON form (a dict) """
    r = Reader(file)
    if not isinstance(model, dict) or 'starsim' not in model:
        r.error('the top level', 'this is not Starsim\'s JSON form, which is a dict with "starsim" (the version that wrote it), the sim\'s settings, "diseases", and other modules')
    unknown = set(model) - {'starsim'} - set(settings) - {'diseases'} - set(module_keys)
    if unknown:
        r.error('the top level', f'{sc.strjoin(sorted(unknown))} is not a setting or a kind of module; the settings are {sc.strjoin(settings)}')
    lines, diseases = [], []
    taken = {'ss', 'ssl', 'np', 'sim'}
    for d in model.get('diseases', []):
        if 'type' in d: # A disease written as code, by its type and parameters
            var = f'disease{len(diseases) + 1}'
            lines += [f'{var} = {r.module(d, "diseases")}', '']
            diseases.append(var)
            continue
        names = {s if isinstance(s, str) else s.get('name') for s in d.get('states', [])} | set(d.get('pars', {})) | set(d.get('transitions', {}))
        var = str(d.get('name', 'disease'))
        n = 0
        while not var.isidentifier() or keyword.iskeyword(var) or var in taken | names:
            n += 1
            var = f'disease{n}'
        taken.add(var)
        lines += r.disease(d, var) + ['']
        diseases.append(var)
    args = [f'diseases=[{", ".join(diseases)}]'] if diseases else []
    for key in module_keys:
        mods = [r.module(m, key) for m in model.get(key, [])]
        if mods:
            args.append(f'{key}=[{", ".join(mods)}]')
    for key in settings:
        if key in model:
            args.append(f'{key}={r.value(model[key], set(), key)}')
    lines.append('sim = ss.Sim(' + ', '.join(args) + ')')
    body = '\n'.join(lines)
    imports = ['import numpy as np'] * ('np.' in body) + ['import starsim as ss'] + ['import starsim.library as ssl'] * ('ssl.' in body) + ['from scipy.special import expit'] * ('expit(' in body)
    header = f'# Read from {sc.path(file).name if file else "a model"} (Starsim {model["starsim"]}\'s JSON form) by ss.import_from()'
    return '\n'.join([header, ''] + imports + [''] + lines + ['', "if __name__ == '__main__':", '    sim.run()', '    sim.plot()', ''])


def load_json(source, verbose=None, **kwargs):
    """
    Read a model in Starsim's JSON form into a sim, by writing its code and running it

    Args:
        source (str/dict): the file, or the model as a dict
        verbose (bool): whether the sim prints as it runs
        kwargs (dict): passed to `ss.Sim()`, after the file's settings (e.g. `n_agents=`)

    Returns:
        An `ss.Sim` (not yet initialized), with its code in `sim.imported.code`
    """
    file = None
    if isinstance(source, dict):
        model = source
    else:
        file = source
        model = sc.loadjson(source)
    text = code(model, file=file)
    space = dict(__name__='imported')
    exec(compile(text, f'<{file or "model"}>', 'exec'), space)
    sim = space['sim']
    if kwargs or verbose is not None:
        sim = sim.new(**kwargs, **({} if verbose is None else dict(verbose=verbose)))
    sim.imported = sc.obj(format='json', file=file, choices=[], incomplete=[], code=text, prompt=None, warned=False)
    return sim
