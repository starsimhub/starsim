"""
Import a SimInf model (an R script, or its pieces as Python data)

SimInf (https://github.com/stewid/SimInf) writes a model as transitions in `mparse()`
("S -> beta*S*I/(S+I+R) -> I"), or as one of its predefined models (`SIR()`, `SIS()`,
`SEIR()`, `SISe()`, `SISe3()`), each run in many nodes (herds) linked by scheduled
events. The transitions become a disease, the nodes a stratification of places, and
the events a movement register (`ss.geo.Mobility(data=...)`); the continuous state of the
predefined SISe models (the environmental infectious pressure `phi`) becomes a state
that is not people. The R is read, not run: what the script computes (e.g. `u0_SIR()`
or a data file) can be passed in instead, e.g. `load_siminf('model.R', u0='u0.csv',
events='events.csv')`, and C code (`pts_fun`) is listed as not translated.

Used with: compartments; declared modules
"""
import re
import ast
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from .core import Source, build

__all__ = ['load_siminf']


#%% Reading R

class Unknown:
    """ An R expression that cannot be read without running R """
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return f'<computed in R: {self.text}>'


tokens_re = re.compile(r'''
    (?P<comment>\#[^\n]*) | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*') |
    (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?L?) | (?P<name>[A-Za-z_.][A-Za-z0-9_.]*) |
    (?P<op><-|<=|>=|==|!=|&&|\|\||%[^%]*%|[-+*/^(),\[\]{}$=<>!:~;&|@]) | (?P<nl>\n) | (?P<ws>[ \t\r]+)
''', re.VERBOSE)


def tokenize(text):
    """ R source as (kind, value) tokens, without comments or spaces """
    out = []
    for m in tokens_re.finditer(text):
        kind = m.lastgroup
        if kind not in ['comment', 'ws']:
            out.append((kind, m.group()))
    return out


class RParser:
    """ A small parser for the R that defines a model: assignments, calls, arithmetic, indexing, and $ """
    binary = [['<-', '='], ['~'], ['|', '||'], ['&', '&&'], ['==', '!=', '<', '>', '<=', '>='], ['+', '-'], ['*', '/'], ['%'], [':'], ['^']]

    def __init__(self, text):
        self.toks = tokenize(text)
        self.i = 0

    def peek(self, skip_nl=False):
        i = self.i
        while skip_nl and i < len(self.toks) and self.toks[i][0] == 'nl':
            i += 1
        return self.toks[i] if i < len(self.toks) else (None, None)

    def take(self):
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def skip_nl(self):
        while self.peek()[0] == 'nl' or self.peek()[1] == ';':
            self.i += 1

    def statements(self):
        """ The top-level statements, as (tree, source text) """
        out = []
        self.skip_nl()
        while self.i < len(self.toks):
            start = self.i
            try:
                tree = self.expr(0, nl=False)
            except (SyntaxError, IndexError): # A statement that cannot be read is skipped, to the end of its line
                self.i = start
                while self.i < len(self.toks) and self.toks[self.i][0] != 'nl':
                    self.i += 1
                tree = ('unread',)
            out.append((tree, ' '.join(v for k,v in self.toks[start:self.i] if k != 'nl')))
            self.skip_nl()
        return out

    def expr(self, level, nl=True):
        """ An expression at a level of precedence; inside brackets, newlines do not end it """
        if level == len(self.binary):
            return self.unary(nl)
        left = self.expr(level + 1, nl)
        while True:
            kind, val = self.peek(skip_nl=nl)
            ops = self.binary[level]
            if val is None or not (val in ops or (ops == ['%'] and kind == 'op' and val.startswith('%'))):
                return left
            if nl:
                self.skip_nl()
            self.take()
            if nl:
                self.skip_nl()
            right = self.expr(level if val in ['<-', '=', '^'] else level + 1, nl) # Right-associative
            left = ('op', val, left, right)

    def unary(self, nl):
        kind, val = self.peek()
        if val in ['-', '+', '!']:
            self.take()
            return ('unary', val, self.unary(nl))
        return self.postfix(self.atom(), nl)

    def atom(self):
        kind, val = self.take()
        if kind == 'num':
            return ('num', float(val.rstrip('L')))
        elif kind == 'str':
            return ('str', ast.literal_eval(val))
        elif kind == 'name':
            if val in ['TRUE', 'FALSE']:
                return ('num', float(val == 'TRUE'))
            elif val in ['NULL', 'NA']:
                return ('null',)
            elif val == 'function': # Not read: skip its arguments and body
                depth = 0
                while True:
                    k, v = self.take()
                    depth += v in ['(', '{']
                    depth -= v in [')', '}']
                    if depth == 0 and v in [')', '}'] and self.peek()[1] != '{':
                        return ('function',)
            return ('name', val)
        elif val == '(':
            self.skip_nl()
            out = self.expr(0)
            self.skip_nl()
            self.take()
            return out
        elif val == '{':
            body = []
            self.skip_nl()
            while self.peek()[1] != '}':
                body.append(self.expr(0, nl=False))
                self.skip_nl()
            self.take()
            return ('block', body)
        errormsg = f'Cannot read the R near "{val}"'
        raise SyntaxError(errormsg)

    def postfix(self, node, nl):
        while True:
            kind, val = self.peek()
            if val == '(':
                self.take()
                node = ('call', node, self.args(')'))
            elif val == '[':
                self.take()
                double = self.peek()[1] == '['
                if double:
                    self.take()
                node = ('index', node, self.args(']'))
                if double:
                    self.take()
            elif val in ['$', '@']:
                self.take()
                node = ('dollar', node, self.take()[1])
            else:
                return node

    def args(self, close):
        """ The arguments of a call or an index, as (name or None, tree) """
        out = []
        self.skip_nl()
        while self.peek()[1] != close:
            if self.peek()[1] == ',': # An empty argument, as in x[, 1]
                out.append((None, ('empty',)))
                self.take()
                self.skip_nl()
                continue
            name = None
            if self.peek()[0] in ['name', 'str'] and self.i + 1 < len(self.toks) and self.toks[self.i + 1][1] == '=':
                name = self.take()[1].strip('"\'')
                self.take()
                self.skip_nl()
            out.append((name, self.expr(1)))
            self.skip_nl()
            if self.peek()[1] == ',':
                self.take()
                self.skip_nl()
        self.take()
        return out


def unparse(node):
    """ An R tree as text, for messages """
    kind = node[0]
    if kind == 'num':
        return f'{node[1]:g}'
    elif kind == 'str':
        return repr(node[1])
    elif kind == 'name':
        return node[1]
    elif kind == 'op':
        return f'{unparse(node[2])} {node[1]} {unparse(node[3])}'
    elif kind == 'unary':
        return f'{node[1]}{unparse(node[2])}'
    elif kind == 'call':
        return f'{unparse(node[1])}(' + ', '.join((f'{k} = ' if k else '') + unparse(v) for k,v in node[2]) + ')'
    elif kind == 'index':
        return f'{unparse(node[1])}[' + ', '.join(unparse(v) for k,v in node[2]) + ']'
    elif kind == 'dollar':
        return f'{unparse(node[1])}${node[2]}'
    return kind


class REval:
    """ Work out the value of an R expression where it is literal (numbers, strings, c(), seq(), rep(), data.frame(), ...), or an Unknown """
    def __init__(self):
        self.env = {}

    def __call__(self, node):
        try:
            return self.eval(node)
        except Exception:
            return Unknown(unparse(node))

    def eval(self, node):
        kind = node[0]
        ev = self.eval
        if kind in ['num', 'str']:
            return node[1]
        elif kind == 'null':
            return None
        elif kind == 'name':
            val = self.env.get(node[1], Unknown(node[1]))
            if isinstance(val, Unknown):
                raise ValueError(val.text)
            return val
        elif kind == 'unary':
            val = np.asarray(ev(node[2]))
            return -val if node[1] == '-' else val if node[1] == '+' else ~val.astype(bool)
        elif kind == 'op':
            op, a, b = node[1], ev(node[2]), ev(node[3])
            if op == ':':
                return np.arange(a, b + np.sign(b - a or 1), np.sign(b - a or 1)).astype(float)
            a, b = np.asarray(a), np.asarray(b)
            ops = {'+':np.add, '-':np.subtract, '*':np.multiply, '/':np.divide, '^':np.power, '==':np.equal, '!=':np.not_equal,
                   '<':np.less, '>':np.greater, '<=':np.less_equal, '>=':np.greater_equal}
            return ops[op](a, b)
        elif kind == 'dollar':
            obj = ev(node[1])
            return obj[node[2]]
        elif kind == 'index':
            obj = ev(node[1])
            idx = [ev(v) for k,v in node[2] if v[0] != 'empty']
            if isinstance(obj, pd.DataFrame) and len(node[2]) == 2 and node[2][0][1][0] == 'empty': # df[, cols]
                return obj[list(sc.tolist(ev(node[2][1][1])))]
            if len(idx) == 1:
                return np.asarray(obj)[np.asarray(idx[0], dtype=int) - 1]
            raise ValueError('indexing')
        elif kind == 'call':
            return self.call(node)
        raise ValueError(kind)

    def call(self, node):
        func = unparse(node[1])
        if func in ['rep', 'rep.int', 'rep_len']: # The length may be computed
            pos = [self(v) for k,v in node[2] if k is None]
            pos = [p for p in pos if not isinstance(p, Unknown)] if not isinstance(pos[0], Unknown) else pos
            kw = {k:self(v) for k,v in node[2] if k is not None}
            if isinstance(pos[0], Unknown) or any(isinstance(v, Unknown) for v in kw.values()):
                raise ValueError('rep')
        else:
            pos = [self.eval(v) for k,v in node[2] if k is None]
            kw = {k:self.eval(v) for k,v in node[2] if k is not None}
        if func == 'c':
            vals = pos + list(kw.values())
            if all(isinstance(v, str) for v in vals):
                return vals if not kw else dict(kw)
            if kw and not pos: # A named vector, as SimInf's gdata
                return {k:float(v) for k,v in kw.items()}
            return np.concatenate([np.atleast_1d(np.asarray(v, dtype=float)) for v in vals])
        elif func == 'seq':
            args = dict(zip(['from', 'to', 'by'], pos))
            args.update(kw)
            frm, to = args.get('from', 1), args.get('to', 1)
            by = args.get('by', 1)
            if 'length.out' in args:
                return np.linspace(frm, to, int(args['length.out']))
            return np.arange(frm, to + by/2, by).astype(float)
        elif func in ['rep', 'rep.int', 'rep_len']:
            if len(node[2]) > 1 and isinstance(self(node[2][1][1]), Unknown) and np.size(pos[0] if pos else 0) == 1: # rep(0, nrow(u0)): the same value everywhere
                return float(pos[0])
            x = np.atleast_1d(np.asarray(pos[0]))
            times = kw.get('times', pos[1] if len(pos) > 1 else 1)
            if 'each' in kw:
                x = np.repeat(x, int(kw['each']))
            if 'length.out' in kw:
                return np.resize(x, int(kw['length.out']))
            return np.tile(x, int(times)) if np.isscalar(times) or np.size(times) == 1 else np.repeat(x, np.asarray(times, dtype=int))
        elif func == 'nrow':
            return len(pos[0])
        elif func in ['length', 'sum', 'exp', 'log', 'sqrt', 'abs', 'round']:
            f = dict(length=len, sum=np.sum, exp=np.exp, log=np.log, sqrt=np.sqrt, abs=np.abs, round=np.round)[func]
            return f(*pos)
        elif func in ['data.frame', 'list']:
            cols = dict(kw)
            if func == 'data.frame':
                n = max([np.size(v) for v in cols.values()] or [0])
                return pd.DataFrame({k:np.resize(np.asarray(v), n) if not isinstance(v, str) else [v]*n for k,v in cols.items()})
            return cols
        elif func in ['as.integer', 'as.numeric', 'as.double']:
            return np.asarray(pos[0], dtype=float)
        elif func in ['read.csv', 'read.table']:
            return pd.read_csv(pos[0] if pos else kw['file'])
        raise ValueError(func)

    def assign(self, target, value):
        """ x <- value, x$col <- value, and x$col[i] <- value """
        kind = target[0]
        if kind == 'name':
            self.env[target[1]] = value
            return
        if kind == 'dollar' and target[1][0] == 'name':
            obj = self.env.get(target[1][1])
            if isinstance(obj, (pd.DataFrame, dict)):
                obj[target[2]] = value
                return
        if kind == 'index' and target[1][0] == 'dollar' and target[1][1][0] == 'name':
            obj = self.env.get(target[1][1][1])
            if isinstance(obj, pd.DataFrame):
                idx = np.asarray(self.eval(target[2][0][1]), dtype=int) - 1
                col = obj[target[1][2]].values.copy()
                col[idx] = value
                obj[target[1][2]] = col
                return
        raise ValueError(f'cannot assign to {unparse(target)}')


#%% SimInf's models

predefined = dict(
    SIR = dict(compartments=['S', 'I', 'R'], transitions=['S -> beta*S*I/(S+I+R) -> I', 'I -> gamma*I -> R'], ldata=['beta', 'gamma'],
               E=[[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]),
    SIS = dict(compartments=['S', 'I'], transitions=['S -> beta*S*I/(S+I) -> I', 'I -> gamma*I -> S'], ldata=['beta', 'gamma'],
               E=[[1, 1], [0, 1]]),
    SEIR = dict(compartments=['S', 'E', 'I', 'R'], transitions=['S -> beta*S*I/(S+E+I+R) -> E', 'E -> epsilon*E -> I', 'I -> gamma*I -> R'],
                ldata=['beta', 'epsilon', 'gamma'], E=[[1, 1], [0, 1], [0, 1], [0, 1]]),
    SISe = dict(compartments=['S', 'I'], transitions=['S -> upsilon*phi*S -> I', 'I -> gamma*I -> S'], v0=['phi'], E=[[1, 1], [0, 1]],
                gdata=['upsilon', 'gamma', 'alpha', 'beta_t1', 'beta_t2', 'beta_t3', 'beta_t4', 'epsilon'], ldata=['end_t1', 'end_t2', 'end_t3', 'end_t4']),
    SISe3 = dict(compartments=['S_1', 'I_1', 'S_2', 'I_2', 'S_3', 'I_3'],
                 transitions=[f'{a}_{k} -> {r}_{k}*{x}{a}_{k} -> {b}_{k}' for k in [1, 2, 3] for a,r,x,b in [('S', 'upsilon', 'phi*', 'I'), ('I', 'gamma', '', 'S')]],
                 v0=['phi'], gdata=['upsilon_1', 'upsilon_2', 'upsilon_3', 'gamma_1', 'gamma_2', 'gamma_3', 'alpha', 'beta_t1', 'beta_t2', 'beta_t3', 'beta_t4', 'epsilon'],
                 ldata=['end_t1', 'end_t2', 'end_t3', 'end_t4'],
                 E=[[1, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 1]],
                 N=[[2, 0, 0], [2, 0, -1], [0, 2, 0], [0, 2, -1], [0, 0, 0], [0, 0, -1]]),
)
model_args = ['transitions', 'compartments', 'ldata', 'gdata', 'u0', 'v0', 'tspan', 'events', 'E', 'N', 'pts_fun', 'pre_code', 'use_enum']


def c_expr(text):
    """ A propensity (C) in Python syntax: pow() as a power, fmin/fmax as min/max, and casts removed """
    text = re.sub(r'\((int|double)\)', '', text)
    node = ast.parse(text.strip(), mode='eval').body
    def visit(node):
        for field,val in ast.iter_fields(node):
            if isinstance(val, ast.AST):
                setattr(node, field, visit(val))
            elif isinstance(val, list):
                setattr(node, field, [visit(v) if isinstance(v, ast.AST) else v for v in val])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == 'pow' and len(node.args) == 2:
                return ast.BinOp(node.args[0], ast.Pow(), node.args[1])
            node.func.id = dict(fmin='min', fmax='max').get(node.func.id, node.func.id)
        return node
    return ast.unparse(visit(node))


def compartment_change(side, compartments):
    """ The number of each compartment on one side of a transition, e.g. "S + 2*I", "@" for none """
    out = dict.fromkeys(compartments, 0)
    for part in side.replace(' ', '').split('+'):
        if part in ['@', '']:
            continue
        m = re.match(r'^(\d+)\*(.+)$', part)
        n, name = (int(m.group(1)), m.group(2)) if m else (1, part)
        if name not in out:
            errormsg = f'The transition uses "{name}", which is not one of the compartments {sc.strjoin(compartments)}'
            raise ValueError(errormsg)
        out[name] += n
    return out


def as_frame(val):
    """ A table given as a data frame, a CSV file, a dict of columns, or a list of rows """
    if val is None or isinstance(val, pd.DataFrame):
        return val
    if isinstance(val, (str, sc.path().__class__)):
        return pd.read_csv(val)
    return pd.DataFrame(val)


def as_matrix(val, rows):
    """ A select or shift matrix (rows are compartments), from a list of rows, an array, a data frame, or a CSV file """
    if val is None:
        return None
    if isinstance(val, (str, sc.path().__class__)):
        val = pd.read_csv(val, index_col=0)
    if isinstance(val, pd.DataFrame):
        val = val.values
    out = np.asarray(val, dtype=float)
    if out.ndim == 1:
        out = out.reshape(len(rows), -1, order='F') # As R's matrix(c(...), nrow=...), by column
    return out


def seasons(ends, betas):
    """ SimInf's daily decay of phi by quarter of the year (SimInf_forward_euler_linear_decay), as {first day: decay}, for ends in order """
    ends = [int(e) for e in ends]
    if not all(a < b for a,b in zip(ends, ends[1:])):
        return None
    out = {0:betas[0], ends[0]:betas[1], ends[1]:betas[2], ends[2]:betas[3]}
    if ends[3] < 365:
        out[ends[3]] = betas[0]
    return out


#%% Loading

def read_script(filename):
    """ The arguments of the model in an R script, read where they are literal """
    text = sc.loadtext(filename)
    reader = REval()
    found = None
    patches = {}
    for tree,text in RParser(text).statements():
        call = tree
        target = None
        if tree[0] == 'op' and tree[1] in ['<-', '=']:
            target, call = tree[2], tree[3]
        if call[0] == 'call' and unparse(call[1]) in ['mparse', *predefined]:
            found = (unparse(call[1]), call[2])
            if target is not None:
                reader.env[unparse(target)] = Unknown('the model')
            continue
        if target is not None and target[0] == 'index' and target[1][0] == 'dollar' and target[1][1][0] == 'name': # x$col[i] <- value, kept to apply to an x passed in
            idx, value = reader(target[2][0][1]), reader(call)
            if not isinstance(idx, Unknown) and not isinstance(value, Unknown):
                patches.setdefault(target[1][1][1], []).append((target[1][2], np.asarray(idx, dtype=int) - 1, value))
        if target is not None:
            value = reader(call)
            try:
                reader.assign(target, value)
            except Exception:
                if target[0] == 'name':
                    reader.env[target[1]] = Unknown(unparse(call))
    if found is None:
        errormsg = f'"{filename}" has no call to mparse() or to one of SimInf\'s models ({sc.strjoin(predefined)})'
        raise ValueError(errormsg)
    kind, args = found
    names = (model_args if kind == 'mparse' else ['u0', 'tspan', 'events'])
    out = dict(model=kind)
    for i,(key,val) in enumerate(args):
        key = key or (names[i] if i < len(names) else None)
        if key is None:
            continue
        if key == 'u0' and val[0] == 'name':
            out['_patches'] = patches.get(val[1], [])
        val = reader(val)
        if val is not None:
            out[key] = val
    out['_env'] = reader.env
    return out


def load_siminf(source, u0=None, events=None, verbose=None, **kwargs):
    """
    Read a SimInf model (an R script, or its pieces as a dict) into a sim

    The transitions become a disease run with the exact (Gillespie) solver, and several
    nodes become a stratification `node` of places, with the scheduled events as a
    movement register. SimInf's numbers are used as they are, in days.

    Args:
        source (str/dict): an R script that calls `mparse()` or one of SimInf's models, or a dict of its arguments (`transitions`, `compartments`, `u0`, `gdata`, `ldata`, `v0`, `tspan`, `events`, `E`, `N`, and `model` for a predefined model, e.g. 'SIR' with `beta` and `gamma`)
        u0 (DataFrame/str): the initial numbers (a row for each node), if the script computes them (a data frame, CSV file, or dict of columns)
        events (DataFrame/str): the scheduled events, if the script computes them (columns event, time, node, dest, n, proportion, select, shift)
        verbose (bool): whether to print how each transition was read
        kwargs (dict): other arguments of the model to use instead of what the script gives (e.g. `tspan`, `beta`), or passed to `ss.Sim()` (e.g. `stop`, `rand_seed`)

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated
    """
    if isinstance(source, dict):
        args = sc.dcp(source)
        name, file = args.pop('name', 'siminf'), None
    else:
        args = read_script(source)
        name, file = sc.path(source).stem, str(source)
    src = Source(name, format='SimInf', file=file)
    env = args.pop('_env', {})
    kind = args.pop('model', 'mparse')
    simkw = {k:kwargs.pop(k) for k in list(kwargs) if k in ['stop', 'start', 'rand_seed', 'solver', 'dt', 'unit', 'n_agents', 'total_pop']}
    args.update(kwargs)
    if u0 is not None:
        args['u0'] = u0
    if events is not None:
        args['events'] = events
    for key,val in list(args.items()):
        if isinstance(val, Unknown):
            if key in ['u0', 'events']:
                src.skip(key, f'computed in R ({val.text}); pass {key}= to load_siminf() (e.g. a CSV written by write.csv() in R)')
            else:
                src.skip(key, f'computed in R ({val.text}); pass {key}= to load_siminf()')
            del args[key]

    # The model's definition
    if kind in predefined:
        pre = predefined[kind]
        defn = dict(compartments=pre['compartments'], transitions=pre['transitions'], E=pre.get('E'), N=pre.get('N'))
        gdata = {k:args[k] for k in pre.get('gdata', []) if k in args}
        ldata = {k:args[k] for k in pre.get('ldata', []) if k in args}
        v0 = {k:args.get(k, 0) for k in pre.get('v0', [])}
    else:
        defn = dict(compartments=args.get('compartments'), transitions=args.get('transitions'), E=args.get('E'), N=args.get('N'))
        gdata = args.get('gdata') or {}
        ldata = args.get('ldata') or {}
        v0 = args.get('v0') or {}
        for key in ['pts_fun', 'pre_code']:
            if args.get(key):
                src.skip(key, f'C code ({sc.strjoin(sc.tolist(args[key]))[:120]}) is not translated: the continuous state it updates ({sc.strjoin(v0) or "none"}) stays at its initial value; write its update as transitions of a state that is not people')
    if isinstance(gdata, pd.DataFrame):
        gdata = {k:gdata[k].values for k in gdata.columns}
    if isinstance(ldata, pd.DataFrame):
        ldata = {k:ldata[k].values for k in ldata.columns}
    if isinstance(v0, pd.DataFrame):
        v0 = {k:v0[k].values for k in v0.columns}
    compartments = defn['compartments']
    if compartments is None or defn['transitions'] is None:
        errormsg = f'The SimInf model "{name}" needs its compartments and transitions (as literal c() vectors, or passed in)'
        raise ValueError(errormsg)
    compartments = list(compartments)

    # Initial numbers, by node
    u0 = as_frame(args.get('u0'))
    if isinstance(u0, pd.DataFrame):
        u0 = u0[[c for c in compartments if c in u0.columns]].copy()
    elif u0 is not None:
        u0 = pd.DataFrame([np.atleast_1d(u0)], columns=compartments)
    n_nodes = len(u0) if u0 is not None else 1
    for col,idx,val in args.get('_patches', []): # Assignments to u0 in the script (u0$I[1] <- 1), which also apply to a u0 that was passed in
        if u0 is not None and col in u0:
            vals = u0[col].values.copy()
            vals[idx] = val
            u0[col] = vals

    # States: the compartments, and the continuous states
    src.declared = True # Every compartment is people, including one joined to the others only by events (aging)
    for c in compartments: # With several nodes, the initial numbers by node (the rest in the first compartment)
        init = float(u0[c].sum()) if u0 is not None and c in u0 and n_nodes == 1 else 0
        if n_nodes > 1 and c in u0 and c != u0.columns[0] and u0[c].any():
            vals = u0[c].values
            init = ss.by(node=vals.astype(int) if np.all(vals == np.round(vals)) else vals.astype(float))
        src.add_state(c, init=init)
    for k,val in v0.items():
        src.add_state(k, people=False, init=None)

    # Parameters: the same in every node, or by node
    def add(key, val):
        arr = np.atleast_1d(np.asarray(val, dtype=float))
        if len(np.unique(arr)) == 1:
            src.add_par(key, float(arr[0]))
        elif len(arr) == n_nodes:
            src.add_par(key, ss.by(node=arr))
        else:
            src.skip(key, f'{len(arr)} values for {n_nodes} nodes')
    seasonal = ['end_t1', 'end_t2', 'end_t3', 'end_t4', 'beta_t1', 'beta_t2', 'beta_t3', 'beta_t4', 'alpha', 'epsilon'] if kind in ['SISe', 'SISe3'] else [] # Used for phi, below
    for key,val in {**gdata, **ldata}.items():
        if key not in seasonal:
            add(key, val)

    # Transitions: variables (N <- S+I+R) are named expressions; the rest are flows
    for i,t in enumerate(sc.tolist(defn['transitions'])):
        if '<-' in t and '->' not in t:
            var, expr = t.split('<-', 1)
            src.exprs[re.sub(r'\(\w+\)', '', var).strip()] = c_expr(expr)
            continue
        parts = t.split('->')
        if len(parts) < 3:
            src.skip(f'transition {t!r}', 'not of the form "from -> propensity -> to"')
            continue
        frm, to = compartment_change(parts[0], compartments), compartment_change(parts[-1], compartments)
        rate = c_expr('->'.join(parts[1:-1]))
        change = {c:to[c] - frm[c] for c in compartments if to[c] != frm[c]}
        lost = [c for c,n in change.items() if n == -1]
        gained = [c for c,n in change.items() if n == 1]
        if any(abs(n) > 1 for n in change.values()) or len(lost) > 1 or len(gained) > 1 or not change:
            src.skip(f'transition {t!r}', f'changes the compartments by {change}, which is not one person moving between two of them')
            continue
        src.add_flow(lost[0] if lost else None, gained[0] if gained else None, rate, per='total')

    # The continuous state of SISe and SISe3: phi decays by quarter of the year, and the infected shed into it, once a day
    if kind in ['SISe', 'SISe3']:
        ends = [np.atleast_1d(args.get(f'end_t{k}', np.nan))[0] for k in range(1, 5)]
        betas = [float(np.atleast_1d(args.get(f'beta_t{k}', np.nan))[0]) for k in range(1, 5)]
        season = seasons(ends, betas) if not np.isnan(ends + betas).any() else None
        alpha, epsilon = float(np.atleast_1d(args.get('alpha', 0))[0]), float(np.atleast_1d(args.get('epsilon', 0))[0])
        if season is None:
            src.skip('phi', f'the decay of phi by quarter needs end_t1 < end_t2 < end_t3 < end_t4 and beta_t1 to beta_t4, as numbers (got {ends}, {betas})')
        else:
            # SimInf's daily update phi*(1 - b) + alpha*I/n + epsilon, as the continuous rates that give the same phi at the end of each day
            decay = {day:-np.log(1 - b) for day,b in season.items()}
            gain = {day:(decay[day]/b if b > 0 else 1.0) for day,b in season.items()}
            src.add_par('phi_decay', ss.timeseries({day:ss.perday(r) for day,r in decay.items()}, period=ss.years(1)), doc='the daily decay of phi, by season, as a rate')
            src.add_par('phi_shed', ss.timeseries({day:ss.perday(alpha*g) for day,g in gain.items()}, period=ss.years(1)), doc='the shedding of the infected into phi, as a rate')
            infected = '+'.join(c for c in compartments if c.startswith('I'))
            everyone = '+'.join(compartments)
            src.add_flow(None, 'phi', f'phi_shed * ({infected}) / ({everyone})', name='shedding', label='shedding into phi')
            if epsilon:
                src.add_par('phi_background', ss.timeseries({day:ss.perday(epsilon*g) for day,g in gain.items()}, period=ss.years(1)), doc='the background infectious pressure')
                src.add_flow(None, 'phi', 'phi_background', name='background')
            src.add_flow('phi', None, 'phi_decay', name='decay', label='decay of phi')
            phi0 = np.atleast_1d(args.get('phi', 0)).astype(float)
            if phi0.any():
                src.skip('phi', f'a nonzero initial phi ({phi0[:5]}...) is not translated; phi starts at 0')

    # Nodes as places, with the events as a movement register
    ev = as_frame(args.get('events'))
    register = None
    if ev is not None and len(ev):
        register, notes = event_register(ev, compartments, as_matrix(defn['E'], compartments), as_matrix(defn['N'], compartments))
        for key,why in notes:
            src.skip(key, why)
    if n_nodes > 1 or register is not None:
        size = u0.sum(axis=1).values if u0 is not None else np.ones(1)
        strat = dict(node=[str(i + 1) for i in range(n_nodes)], kind='space', split=size/size.sum() if size.sum() else None)
        if v0:
            strat['states'] = compartments + list(v0)
        if register is not None:
            strat['mobility'] = ss.geo.Mobility(data=register)
        src.stratify(**strat)

    # Time: SimInf reports at tspan, from the first time
    tspan = args.get('tspan')
    cfg = dict(solver='gillespie', unit='day')
    if tspan is not None:
        tspan = np.atleast_1d(np.asarray(tspan, dtype=float))
        cfg.update(start=float(tspan[0] - 1), stop=float(tspan[-1]))
    else:
        src.skip('tspan', 'not given: the sim runs for Starsim\'s default time')
    cfg.update(simkw)
    cfg.setdefault('total_pop', float(u0.values.sum()) if u0 is not None else 0)
    return build(src, verbose=verbose, **cfg)


def event_register(ev, compartments, E, N):
    """ SimInf's scheduled events as a movement register for ss.geo.Mobility, and what could not be translated """
    notes = []
    ev = ev.copy()
    for col,default in [('dest', 0), ('proportion', 0), ('shift', 0)]:
        if col not in ev:
            ev[col] = default
    ev['event'] = ev['event'].replace({0:'exit', 1:'enter', 2:'intTrans', 3:'extTrans'})
    if ((ev.n == 0) & (ev.proportion > 0)).any():
        k = ((ev.n == 0) & (ev.proportion > 0)).sum()
        notes.append(('events', f'{k} events give a proportion rather than a number of individuals, which a register cannot; they were left out'))
        ev = ev[ev.n > 0]
    if E is None:
        notes.append(('events', 'no select matrix (E): every event chooses from every compartment'))
    def selected(col):
        if E is None or pd.isna(col) or int(col) < 1 or int(col) > E.shape[1]:
            return None
        return [c for c,v in zip(compartments, E[:, int(col) - 1]) if v > 0]
    def shifted(select, shift):
        """ Where an internal transfer moves the compartments it selects, by a column of the shift matrix """
        if int(shift) < 1:
            return None
        sel = selected(select) or compartments
        return ','.join(compartments[compartments.index(c) + int(N[compartments.index(c), int(shift) - 1])] for c in sel)
    sel = {s:selected(s) for s in ev.select.unique()}
    sel = {s:(None if v is not None and len(v) == len(compartments) else v) for s,v in sel.items()} # Every compartment is the register's default (naming them all gives different results: see the report)
    states = ev.select.map(lambda s: ','.join(sel[s]) if sel[s] else None)
    first = ev.select.map(lambda s: sel[s][0] if sel[s] else compartments[0]) # Entering individuals go to the first compartment selected
    states = states.where(ev.event != 'enter', first)
    is_int = ev.event == 'intTrans'
    to_states = pd.Series([None]*len(ev), index=ev.index, dtype=object)
    if is_int.any():
        if N is None:
            notes.append(('events', f'{is_int.sum()} internal transfers (intTrans) need a shift matrix (N); they were left out'))
            ev, states, to_states = ev[~is_int], states[~is_int], to_states[~is_int]
        else:
            pairs = list(zip(ev.select[is_int], ev['shift'][is_int]))
            found = {pair:shifted(*pair) for pair in set(pairs)} # Each pair of a select and a shift column once, since there can be many events
            to_states[is_int] = [found[pair] for pair in pairs]
    register = pd.DataFrame({
        'date':      ev.time.values,
        'from':      ev.node.where(ev.event != 'enter').values,
        'to':        ev.node.where(ev.event.isin(['enter', 'intTrans']), ev.dest.where(ev.event == 'extTrans')).values,
        'n':         ev.n.values,
        'states':    states.values,
    })
    if to_states.notna().any():
        register['to_states'] = to_states.values
    return register, notes
