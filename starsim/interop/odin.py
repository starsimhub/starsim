"""
Import an odin model (an R file in odin's DSL)

odin (https://mrc-ide.github.io/odin) writes a model as equations: `deriv(X) <- ...`
for an ODE, or `update(X) <- ...` with `rbinom()` draws for a model in discrete time.
odin2's syntax is read too: its draws (`Binomial()`, `Poisson()`, ...), its built-in
`dt` and `time`, a count over a period (`initial(x, zero_every = k)`), and data with
their likelihoods (`cases <- data()` and `cases ~ Poisson(x)`).
Each equation is split into its signed terms, and a term that leaves one state and
enters another is a flow between them; a term with no partner is an inflow or an
outflow. The file is parsed here, so importing needs neither R nor odin.

Arrays are translated in three cases: arrays of alike, independent copies (such as
odin's replicates), which become one model run many times; arrays over levels with
flows from each level to the next (aging), which become an age stratification; and
arrays coupled by a contact matrix (`sum(s_ij[i, ])`), which become a stratification
with that matrix.
`delay()` of a flow into a state becomes a fixed dwell time in that state. What is
listed as not translated is left for a person, or the import-model skill, to add.
"""
import re
import ast
import math
import keyword
import numpy as np
import sciris as sc
import starsim as ss
from .core import Source, build, factors, product

__all__ = ['load_odin']

lhs_pattern = re.compile(r'^(?:(deriv|update|initial|dim|output|config)\s*\()?\s*([A-Za-z_][\w.]*)\s*(?:\[([^\]]*)\])?\s*(?:,\s*zero_every\s*=\s*([^)]+?))?\s*\)?$')
draws = dict(Binomial='rbinom', Poisson='rpois', Multinomial='rmultinom', Normal='rnorm', Uniform='runif', Exponential='rexp', Gamma='rgamma', NegativeBinomial='rnbinom', Hypergeometric='rhyper') # odin2's draws, as odin1's
years = dict(day=365, week=365/7, month=12, year=1) # Units in a year, for age bands
reserved = sorted(set(keyword.kwlist) - {'if', 'else', 'for', 'while', 'break', 'in', 'True', 'False', 'None'}) # Names R allows and Python does not (lambda, is, ...)


#%% Reading the file

def statements(text):
    """ The statements of an odin file as (left, sign, right), with sign '<-' or '~', joining lines that continue, with names Python reserves renamed (lambda is lambda_) """
    out = []
    buf = ''
    for line in text.splitlines():
        line = line.split('#')[0].rstrip()
        line = re.sub(rf'\b({"|".join(reserved)})\b', r'\1_', line)
        if not line.strip():
            continue
        buf = f'{buf} {line.strip()}' if buf else line.strip()
        depth = buf.count('(') - buf.count(')') + buf.count('[') - buf.count(']')
        if depth > 0 or re.search(r'[-+*/^,<~]$|<-$', buf) or '<-' not in buf and '=' not in buf and '~' not in buf:
            continue
        sep = '<-' if '<-' in buf else '~' if '~' in buf else '='
        left, right = buf.split(sep, 1)
        out.append((left.strip(), '~' if sep == '~' else '<-', right.strip()))
        buf = ''
    return out


def r2py(text):
    """ An R expression in Python syntax """
    text = text.replace('^', '**').replace('TRUE', 'True').replace('FALSE', 'False').replace('%%', '%')
    text = text.replace('&&', ' and ').replace('||', ' or ').replace('&', ' and ').replace('|', ' or ')
    text = re.sub(r'!(?!=)', ' not ', text)
    text = re.sub(r'(\w)\[\]', r'\1', text) # X[] is the whole array
    return r_if(text)


def closing(text, start):
    """ The index of the bracket that closes the one at text[start] """
    depth = 0
    for j in range(start, len(text)):
        depth += text[j] in '(['
        depth -= text[j] in ')]'
        if depth == 0:
            return j
    errormsg = f'Unbalanced brackets in "{text}"'
    raise SyntaxError(errormsg)


def r_if(text):
    """ R's `if (c) a else b` as Python's `((a) if (c) else (b))`; `b` runs to the end of the brackets the if is in """
    m = re.search(r'\bif\s*\(', text)
    if not m:
        return text
    j = closing(text, m.end() - 1)
    cond, rest = text[m.end():j], text[j+1:]
    depth = 0
    for k,c in enumerate(rest): # The else at the same depth
        depth += c in '(['
        depth -= c in ')]'
        if depth == 0 and re.match(r'else\b', rest[k:]) and (k == 0 or not (rest[k-1].isalnum() or rest[k-1] == '_')):
            break
    else:
        errormsg = f'An if with no else in "{text}"'
        raise SyntaxError(errormsg)
    a, other = rest[:k], rest[k+4:]
    depth = 0
    for e,c in enumerate(other): # The end of the else: the end of the text, or a bracket that closes one opened before the if
        depth += c in '(['
        depth -= c in ')]'
        if depth < 0 or (depth == 0 and c == ','):
            break
    else:
        e = len(other)
    b, after = other[:e], other[e:]
    return f'{text[:m.start()]}(({r_if(a.strip())}) if ({r_if(cond)}) else ({r_if(b.strip())})){r_if(after)}'


def index_range(text, env):
    """ The levels (1-based) an index on the left covers: None for all of them """
    text = (text or '').strip()
    if not text or text == 'i':
        return None
    if ':' in text:
        a, b = text.split(':', 1)
        return list(range(int(evaluate(r2py(a), env)), int(evaluate(r2py(b), env)) + 1))
    return [int(evaluate(r2py(text), env))]


def evaluate(text, env, i=None):
    """ The value of an expression of numbers and parameters (arrays are 1-based), or raise KeyError """
    node = ast.parse(text, mode='eval').body
    fns = dict(exp=np.exp, log=np.log, sqrt=np.sqrt, abs=np.abs, sum=np.sum, min=np.minimum, max=np.maximum, round=np.round, floor=np.floor, ceiling=np.ceil)
    def ev(n):
        if isinstance(n, ast.Constant):
            return n.value
        elif isinstance(n, ast.Name):
            if n.id == 'i' and i is not None:
                return i
            if n.id in ['pi']:
                return math.pi
            return env[n.id]
        elif isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            ops = {ast.Add:np.add, ast.Sub:np.subtract, ast.Mult:np.multiply, ast.Div:np.divide, ast.Pow:np.power}
            return ops[type(n.op)](a, b)
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        elif isinstance(n, ast.Subscript):
            arr = np.atleast_1d(ev(n.value))
            return arr[int(ev(n.slice)) - 1]
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in fns:
            return fns[n.func.id](*[ev(a) for a in n.args])
        raise KeyError(ast.unparse(n))
    return ev(node)


def default(text):
    """ The default of user(x) or parameter(x), or None if it has none; or False if this is not one """
    m = re.match(r'^(user|parameter)\s*\((.*)\)$', text.strip())
    if not m:
        return False
    args = [a.strip() for a in m.group(2).split(',') if a.strip()]
    for a in args:
        if a.startswith('default'):
            return a.split('=', 1)[1].strip()
    if args and '=' not in args[0]:
        return args[0]
    return None


#%% Expressions, by level

def strip_index(node):
    """ An expression with its indices taken out: X[i] is X, X[i - 1] is X__prev, sum(X) is X__sum, X[k] is X__at_k """
    def visit(n):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            name, s = n.value.id, n.slice
            if isinstance(s, ast.Name) and s.id == 'i':
                return ast.Name(name)
            if isinstance(s, ast.BinOp) and isinstance(s.op, ast.Sub) and isinstance(s.left, ast.Name) and s.left.id == 'i' and isinstance(s.right, ast.Constant) and s.right.value == 1:
                return ast.Name(f'{name}__prev')
            if isinstance(s, ast.Constant):
                return ast.Name(f'{name}__at_{s.value}')
            return ast.Name(f'{name}__index') # Anything else: kept, and found later
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'sum' and len(n.args) == 1 and isinstance(n.args[0], ast.Name):
            return ast.Name(f'{n.args[0].id}__sum')
        for field,val in ast.iter_fields(n):
            if isinstance(val, ast.AST):
                setattr(n, field, visit(val))
            elif isinstance(val, list):
                setattr(n, field, [visit(v) if isinstance(v, ast.AST) else v for v in val])
        return n
    return visit(node)


def signed_terms(node, sign=1):
    """ An expression as a list of (sign, term), with products of sums multiplied out """
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return signed_terms(node.left, sign) + signed_terms(node.right, sign if isinstance(node.op, ast.Add) else -sign)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return signed_terms(node.operand, -sign)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return signed_terms(node.operand, sign)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left, right = signed_terms(node.left), signed_terms(node.right)
        if len(left) > 1 or len(right) > 1:
            return [(sign*a*b, ast.BinOp(x, ast.Mult(), y)) for a,x in left for b,y in right]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = signed_terms(node.left)
        if len(left) > 1:
            return [(sign*a, ast.BinOp(x, ast.Div(), node.right)) for a,x in left]
    return [(sign, node)]


def unsign(sign, term):
    """ A term with any minus signs on its factors moved into its sign: (1, -a * X) is (-1, a * X) """
    num, den = factors(term)
    minus = sum(isinstance(f, ast.UnaryOp) and isinstance(f.op, ast.USub) for f in num + den)
    if not minus:
        return sign, term
    return sign*(-1)**minus, product([strip_sign(f) for f in num], [strip_sign(f) for f in den])


def canonical(node):
    """ A term as text, with its factors in a fixed order, so the same term written two ways matches """
    num, den = factors(node)
    num = sorted((canonical(f) if isinstance(f, ast.BinOp) and isinstance(f.op, (ast.Mult, ast.Div)) else ast.unparse(f) for f in num))
    den = sorted(ast.unparse(f) for f in den)
    num = [f for f in num if f != '1'] or ['1']
    return ' * '.join(num) + ''.join(f' / {f}' for f in den)


#%% The importer

def load_odin(filename, unit='day', stop=None, pars=None, verbose=None, **kwargs):
    """
    Read an odin model (an R file in odin's DSL) into a sim

    Each `deriv()` or `update()` equation is split into signed terms, and the terms
    are paired into flows between states (a term with no partner is an inflow or
    an outflow). A deterministic model (`deriv`) runs as an ODE; one in discrete time
    with `rbinom()` draws runs as a chain binomial, with the probability of each
    draw turned into the rate that gives it over a step. odin has no time unit, so
    the numbers are taken to be in `unit`.

    odin2's syntax is read as odin1's: `Binomial()` is `rbinom()`, `dt` is built in
    (1, unless given in `pars`), a count over a period (`initial(cases, zero_every = 7) <- 0`
    with `update(cases) <- cases + n_SI`) is an outcome of the flows it adds up, and
    data with a likelihood (`deaths <- data()` and `deaths ~ Poisson(rho * cases)`) is
    an outcome named for the data, with that likelihood, which `sim.data` (a column
    `deaths`) is compared with when calibrating.

    Args:
        filename (str): the odin file
        unit (str): the time unit of the model's numbers (odin has none)
        stop (float): when to stop, in that unit (odin is told when it is run)
        pars (dict): values for `user()` parameters, instead of their defaults
        verbose (bool): whether to print how each flow was read
        kwargs (dict): passed to `ss.Sim()`

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated
    """
    text = sc.path(filename).read_text()
    name = sc.path(filename).stem
    src = Source(name, format='odin', file=str(filename))
    src.delays = [] # Each delay() read as a dwell time, to check
    pars = pars or {}
    notes = []

    # Sort the statements
    eqs = dict(deriv={}, update={}, initial={}, dim={}, output={}, config={})
    assign = {} # Name: list of (levels, text)
    order = []
    zero_every = {} # odin2's counts over a period: initial(cases, zero_every = 7) <- 0
    data = [] # odin2's data (cases <- data())
    compare = [] # And their likelihoods: (name, text) for cases ~ Poisson(incidence)
    for left, sign, right in statements(text):
        m = lhs_pattern.match(left.strip())
        if sign == '~':
            compare.append((left, right))
            continue
        if not m:
            src.skip(left, f'could not read the left-hand side of "{left} <- {right}"')
            continue
        kind, var, index, zero = m.groups()
        if zero is not None:
            zero_every[var] = zero.strip()
        if not kind and re.match(r'^data\s*\(\s*\)$', right):
            data.append(var)
        elif kind:
            eqs[kind].setdefault(var, []).append((index, right))
        else:
            assign.setdefault(var, []).append((index, right))
            order.append(var)
    for key in ['output', 'config']:
        for var in eqs[key]:
            src.skip(f'{key}({var})', f'{key}() is not translated (an output is a result to declare by hand)')
    discrete = bool(eqs['update'])
    if discrete and eqs['deriv']:
        src.skip('deriv and update', 'a model with both deriv() and update() equations; only update() was translated')
    state_eqs = eqs['update'] if discrete else eqs['deriv']
    states = list(state_eqs.keys())

    # Parameters and arrays of numbers: worked out in any order
    env = {k:v for k,v in pars.items()}
    if 'dt' not in assign: # odin2's built-in step, 1 unless given in pars (odin1 gives it as dt <- user(1))
        env.setdefault('dt', 1)
    users = {}
    for var,items in assign.items():
        if len(items) == 1 and items[0][0] is None:
            d = default(items[0][1])
            if d is not False:
                users[var] = True
                if var not in env:
                    if d is None:
                        src.skip(var, f'user() parameter with no default; pass it as pars=dict({var}=...)')
                    else:
                        try:
                            env[var] = evaluate(r2py(d), env)
                        except KeyError:
                            assign[var] = [(None, d)]
                            users.pop(var)
    dims = {}
    for i in range(len(assign) + 2):
        for var,items in assign.items():
            if var in env:
                continue
            try:
                if all(idx is None for idx,_ in items) and len(items) == 1:
                    env[var] = evaluate(r2py(items[0][1]), env)
                else: # An array, assigned by level
                    n = int(evaluate(r2py(eqs['dim'][var][0][1]), env))
                    arr = np.full(n, np.nan)
                    for idx,rhs in items:
                        levels = index_range(idx, env) or list(range(1, n + 1))
                        for lev in levels:
                            arr[lev - 1] = evaluate(r2py(rhs), {**env, var:arr}, i=lev)
                    env[var] = arr
            except (KeyError, IndexError, ValueError, TypeError, ZeroDivisionError):
                pass
        for var,items in eqs['dim'].items():
            try:
                dims[var] = int(evaluate(r2py(items[0][1]), env))
            except KeyError:
                pass
    exprs = {var:items for var,items in assign.items() if var not in env}
    dt = env.get('dt', 1)

    # States that are not stocks: the time (update(time) <- (step + 1) * dt), and a count over a period (update(cases) <- if (step %% 7 == 0) ... else cases + ..., or odin2's zero_every)
    counters = {} # Name: (the flows it counts, the period)
    for s in list(state_eqs):
        items = state_eqs[s]
        node = ast.parse(r2py(items[0][1]), mode='eval').body
        if s in zero_every: # initial(cases, zero_every = 7) <- 0, with update(cases) <- cases + n_SI (or deriv(cases) <- n_SI)
            terms = [(sign, term) for sign,term in signed_terms(node) if not (discrete and isinstance(term, ast.Name) and term.id == s)]
            if len(items) == 1 and all(sign > 0 for sign,_ in terms):
                counters[s] = (' + '.join(ast.unparse(term) for _,term in terms), zero_every[s])
            else:
                src.skip(f'{"update" if discrete else "deriv"}({s})', f'"{items[0][1]}" is a count over a period (zero_every) that is not a sum of flows, so it was not translated')
        elif not discrete:
            continue
        elif len(items) == 1 and isinstance(node, ast.IfExp):
            total = ast.unparse(node.body)
            test = node.test
            period = None
            if ast.unparse(node.orelse).replace(' ', '') == f'{s}+{total}'.replace(' ', '') and isinstance(test, ast.Compare) and isinstance(test.left, ast.BinOp) and isinstance(test.left.op, ast.Mod):
                if ast.unparse(test.left.left) == 'step' and ast.unparse(test.comparators[0]) == '0':
                    try:
                        period = float(evaluate(ast.unparse(test.left.right), env))*dt
                    except KeyError:
                        pass
            if period is not None:
                counters[s] = (total, period)
            else:
                src.skip(f'update({s})', f'"{items[0][1]}" is a condition, not {s} plus or minus flows, so it was not translated (a value to declare as a result, or an ss.outcome)')
        elif not any(sign > 0 and isinstance(term, ast.Name) and term.id == s for sign,term in signed_terms(strip_index(node))):
            used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if used <= {'step', 't', 'time', 'dt'}:
                notes.append(f'{s} is the time ("{items[0][1]}"), which the sim keeps itself')
            else:
                src.skip(f'update({s})', f'"{items[0][1]}" is not {s} plus or minus flows, so it was not translated (a value to declare as a result)')
        else:
            continue
        states.remove(s)
        state_eqs.pop(s)
        eqs['initial'].pop(s, None)

    # A contact matrix: Y[i, j] <- M[i, j] * X[j] / N[j], summed over j as sum(Y[i, ]), is exposure to X through ss.contacts()
    mixing = None
    for var,items in list(exprs.items()):
        idx, rhs = items[0]
        if len(items) > 1 or (idx or '').replace(' ', '') not in ['', ',', 'i,j'] or not re.search(r'\[\s*i\s*,\s*j\s*\]', rhs):
            continue
        num, den = factors(ast.parse(r2py(rhs), mode='eval').body)
        mats = [f for f in num if isinstance(f, ast.Subscript) and isinstance(f.value, ast.Name) and np.ndim(env.get(f.value.id)) == 2 and ast.unparse(f.slice).replace(' ', '').strip('()') == 'i,j']
        summed = rf'sum\(\s*{re.escape(var)}\s*\[\s*i\s*,\s*\]\s*\)'
        if len(mats) != 1 or mixing is not None:
            continue
        inner = ast.unparse(product([f for f in num if f is not mats[0]], den))
        if re.search(r'\bi\b', inner.replace('[j]', '')):
            continue
        uses = [(v, k) for v,its in exprs.items() for k,(_,r) in enumerate(its) if re.search(summed, r)]
        if not uses:
            continue
        for v,k in uses:
            exprs[v][k] = (exprs[v][k][0], re.sub(summed, f'({inner.replace("[j]", "[i]")})', exprs[v][k][1]))
        del exprs[var]
        mixing = (mats[0].value.id, np.asarray(env[mats[0].value.id], dtype=float))
        notes.append(f'{var} = {mats[0].value.id}[i, j] * {inner}, summed over j: exposure to {inner.replace("[j]", "")} through the contact matrix {mats[0].value.id} (ss.contacts)')

    # Seasonality: 1 + A * cos(2 * pi * t / 365) (with the time as t, time, or step * dt) is ss.seasonal()
    seasonals = {}
    for var,items in list(exprs.items()):
        node = ast.parse(r2py(items[0][1]), mode='eval').body
        if len(items) > 1 or not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) and ast.unparse(node.left) == '1'):
            continue
        num, den = factors(node.right)
        waves = [f for f in num if isinstance(f, ast.Call) and isinstance(f.func, ast.Name) and f.func.id in ['cos', 'sin'] and len(f.args) == 1]
        if len(waves) != 1:
            continue
        try:
            amp = float(evaluate(ast.unparse(product([f for f in num if f is not waves[0]], den)), env))
            at = lambda k: float(evaluate(ast.unparse(waves[0].args[0]), {**env, 'step':k, 't':k*dt, 'time':k*dt}))
            phase, per_unit = at(0), (at(1) - at(0))/dt
        except (KeyError, TypeError):
            continue
        if unit in years and per_unit > 0 and np.isclose(2*np.pi/per_unit, years[unit], rtol=0.01): # A cycle of a year
            shift = np.pi/2 if waves[0].func.id == 'sin' else 0 # sin(x) peaks a quarter of a cycle later than cos(x)
            peak = float(((shift - phase)/(2*np.pi)) % 1)
            seasonals[var] = ss.seasonal(mean=1.0, amplitude=amp, peak=peak)
            del exprs[var]
            notes.append(f'{var} = {items[0][1]} read as a yearly cycle: ss.seasonal(amplitude={amp:g}, peak={peak:g})')

    # Arrays: alike, independent copies (replicates), or levels with flows between them
    arrayed = [s for s in states if dims.get(s, 1) > 1]
    n_levels = max([dims[s] for s in arrayed], default=1)
    all_text = ' '.join(rhs for items in list(state_eqs.values()) + list(exprs.values()) + list(eqs['initial'].values()) for _,rhs in items)
    by_level = bool(re.search(r'\[\s*i\s*-\s*1\s*\]|\[\s*\d+\s*\]|sum\(', all_text)) or any(idx not in (None, '', 'i') for items in state_eqs.values() for idx,_ in items)
    varying = [k for k,v in env.items() if isinstance(v, np.ndarray) and not np.allclose(v, v[0])]
    copies = arrayed and not by_level and not varying and mixing is None
    dim = 'group' # The name of the dimension the levels are, from what they are counted by: dim(S) <- n_age is age
    if arrayed:
        m = re.match(r'^n_(\w+)$', eqs['dim'][arrayed[0]][0][1].strip())
        dim = m.group(1) if m else dim
    if re.search(r'\[\s*i\s*-\s*1\s*\]', all_text): # Flows from each level to the next: aging (below)
        dim = 'age'
    if copies:
        notes.append(f'the arrays ({sc.strjoin(arrayed)}) are {n_levels} alike, independent copies: imported as one model, to run {n_levels} times with sim.run(n_runs={n_levels})')

    # Named expressions, without their indices
    def py(text):
        return strip_index(ast.parse(r2py(text), mode='eval').body)
    for var,items in exprs.items():
        variants = {ast.unparse(py(rhs)) for _,rhs in items}
        if len(variants) > 1:
            src.skip(var, f'is given differently at different levels ({sc.strjoin(variants)}); the first is used')
        src.exprs[var] = ast.unparse(py(items[0][1]))

    # Initial numbers
    shares = np.zeros(n_levels)
    for s in states:
        items = eqs['initial'].get(s, [])
        vals = np.zeros(n_levels)
        for idx,rhs in items:
            levels = index_range(idx, env) or list(range(1, n_levels + 1))
            for lev in levels:
                try:
                    vals[lev - 1] = evaluate(r2py(rhs), env, i=lev)
                except KeyError as E:
                    src.skip(f'initial({s})', f'the initial value "{rhs}" uses {E}, which is not a number')
        if copies:
            init = vals[0]
        else:
            init = vals.sum()
            shares += vals
        if mixing is not None and init: # Groups coupled only by the contacts, each with its own numbers
            src.add_state(s, init=ss.by(**{dim:list(vals)}))
        else:
            src.add_state(s, init=float(np.round(init, 6)) if init else None) # Rounded, since a sum of shares is not exact

    # Flows: each term of each equation, by state
    terms = {} # canonical text: list of (state, sign, node, levels)
    for s,items in state_eqs.items():
        for idx,rhs in items:
            node = py(rhs)
            levels = index_range(idx, env)
            for sign,term in signed_terms(node):
                sign, term = unsign(sign, term)
                if discrete and isinstance(term, ast.Name) and term.id == s and sign > 0:
                    continue # update(S) <- S - n_SI: the state itself
                terms.setdefault(canonical(term), []).append((s, sign, term, levels))
    seen = {} # Each term once for each state and sign, with the levels of every equation it is in (None for all)
    for key,uses in terms.items():
        merged = {}
        for s,sign,term,levels in uses:
            if (s, sign) in merged:
                old = merged[(s, sign)][3]
                levels = None if old is None or levels is None else sorted(set(old) | set(levels))
            merged[(s, sign)] = (s, sign, term, levels)
        seen[key] = [(s, sign, term, None if levels is not None and set(levels) == set(range(1, n_levels + 1)) else levels) for s,sign,term,levels in merged.values()]

    # Flows between levels of the same state (aging), which make a stratification
    transfers = {}
    for key,uses in list(seen.items()):
        if '__prev' in key:
            base = key.replace('__prev', '')
            partner = [u for u in seen.get(base, []) if u[1] < 0]
            mine = [u for u in uses if u[1] > 0]
            if partner and mine and {u[0] for u in mine} <= {u[0] for u in partner}:
                transfers[base] = sorted({u[0] for u in mine})
                seen[base] = [u for u in seen[base] if not (u[1] < 0 and u[0] in transfers[base])]
                del seen[key]
    if transfers and not copies:
        strat = None
        for base,ss_ in transfers.items():
            num, den = factors(ast.parse(base, mode='eval').body)
            rate = [ast.unparse(f) for f in num + den if ast.unparse(f) not in states]
            vals = env.get(rate[0]) if len(rate) == 1 and not den else None
            if isinstance(vals, np.ndarray) and len(vals) == n_levels and vals[-1] == 0 and np.all(vals[:-1] > 0):
                widths = 1/vals[:-1]/years.get(unit, 1)
                edges = [0] + list(np.round(np.cumsum(widths), 10))
                edges = [int(e) if float(e).is_integer() else float(e) for e in edges]
                strat = dict(age=edges)
                notes.append(f'the flows of {sc.strjoin(ss_)} from each level to the next at {rate[0]} (1/{sc.strjoin(np.round(1/vals[:-1], 6))} {unit}s) read as aging: stratified by age, {edges}')
            else:
                src.skip(f'flows between levels ({base})', f'{sc.strjoin(ss_)} move from each level of the index to the next at "{base}", which is not one rate for each level ending in 0, so it was not read as aging')
        if strat is not None:
            if shares.sum() > 0:
                strat['split'] = list(shares/shares.sum())
            src.stratify(**strat)
        if mixing is not None and strat is not None: # Age groups with a contact matrix between them
            src.config.networks = ss.contacts(age=mixing[1].tolist())
            src.code['networks'] = f'ss.contacts(age={mixing[1].tolist()!r})'
        elif mixing is not None:
            src.skip(f'contact matrix {mixing[0]}', 'the levels it is between are not a stratification, so it was not translated')
    elif mixing is not None:
        levels = [str(lev) for lev in range(1, n_levels + 1)]
        src.stratify(**{dim:levels}, split=list(shares/shares.sum()))
        src.config.networks = ss.contacts(**{dim:mixing[1].tolist()})
        src.code['networks'] = f'ss.contacts({dim}={mixing[1].tolist()!r})'
        notes.append(f'the arrays ({sc.strjoin(arrayed)}) over {n_levels} levels of {dim}, coupled by the contact matrix {mixing[0]}: stratified by {dim} ("1" to "{n_levels}"), with ss.contacts({dim}={mixing[0]})')
    elif arrayed and not copies:
        src.skip('arrays', f'{sc.strjoin(arrayed)} are arrays over {n_levels} levels with no flows between levels: they could be a stratification (stratify()) with values by level (ss.by()), written by hand')

    # Pair the terms: out of one state and into another is a flow; unpaired, an inflow or outflow
    for key,uses in seen.items():
        if not uses: # Only flows between levels, made by the stratification
            continue
        outs = [u for u in uses if u[1] < 0]
        ins = [u for u in uses if u[1] > 0]
        if len(outs) > 1 or len(ins) > 1:
            src.skip(key, f'the term appears in more than one equation with the same sign ({sc.strjoin(u[0] for u in uses)}), so which states it moves people between is not clear')
            continue
        frm = outs[0][0] if outs else None
        to = ins[0][0] if ins else None
        term = (outs or ins)[0][2]
        levels = (ins or outs)[0][3]
        if levels is not None and set(levels) != set(range(1, n_levels + 1)) and not (to and not frm and levels == [1]):
            src.skip(key, f'applies only at levels {levels} of the index ({frm or ""} -> {to or ""})')
            continue
        flow(src, frm, to, term, discrete, env, assign, dt, unit, notes)

    # A count over a period of the flows it adds up is an outcome
    for s,(total,period) in counters.items(): # The period is not needed: an outcome is a flow, which data are compared with as the total since the last data point
        counted = [n for n in re.findall(r'\w+', total) if n != 'sum']
        found = [f.name for f in src.flows if f.name in counted]
        if counted and len(found) == len(counted):
            src.add_outcome(s, source=found)
        else:
            src.skip(f'update({s})', f'a count over a period of {total}, which is not a sum of flows that were translated')

    # odin2's data and their likelihoods: cases ~ Poisson(incidence) is an outcome named for the data, a share of a flow or count, with its likelihood
    used = set()
    for var,rhs in compare:
        var = var.strip()
        why = likelihood(src, var, rhs, data, env, used)
        if why:
            src.skip(f'{var} ~ {rhs}', why)
    for var in data:
        if var not in src.outcomes:
            src.skip(f'{var} (data)', 'data with no likelihood (~) that was translated; give it when calibrating (sim.fit(likelihood=...))')

    # Parameters: those the flows and expressions use
    for f in src.flows:
        if f.rate is not None:
            used |= {n.id for n in ast.walk(ast.parse(f.rate, mode='eval')) if isinstance(n, ast.Name)}
    for e in list(src.exprs.values()):
        used |= {n.id for n in ast.walk(ast.parse(e, mode='eval')) if isinstance(n, ast.Name)}
    for var in sorted(used):
        val = env.get(var)
        if var in seasonals:
            src.add_par(var, seasonals[var])
            continue
        elif val is None:
            continue
        if isinstance(val, np.ndarray):
            if copies or np.allclose(val, val[0]):
                src.add_par(var, float(val[0]))
            elif src.strats:
                src.add_par(var, ss.by(**{dim if mixing is not None else 'age':list(val)}))
            else:
                src.skip(var, f'an array of parameters ({val}) with no stratification to give it')
        else:
            src.add_par(var, float(val) if not isinstance(val, bool) else val)

    # A delay's factor has to be the share who survive it, from the other ways out of the state
    for dname,frm,tau,factor in src.delays:
        rates = []
        for f in [f for f in src.flows if f.src == frm and f.dur is None]:
            try: # The rate for each person, if it is a number
                rates.append(float(evaluate(f.rate, {**env, frm:1} if f.per == 'total' else env)))
            except KeyError:
                rates.append(None)
        expected = math.exp(-sum(rates)*tau) if None not in rates else None
        if expected is None or not np.isclose(factor, expected, rtol=1e-6):
            src.skip(f'{dname} (factor)', f'the factor {factor:.6g} in the delay is not the share who survive {tau:n} in {frm} ({"its other ways out are not all fixed rates" if expected is None else f"{expected:.6g}"}), so the dwell time leaves out part of what the original does')

    # Index names left in the rates are what could not be translated
    for f in list(src.flows):
        found = {n.id for n in ast.walk(ast.parse(f.rate, mode='eval')) if isinstance(n, ast.Name)} if f.rate else set()
        for e in list(found):
            if e in src.exprs:
                found |= {n.id for n in ast.walk(ast.parse(src.exprs[e], mode='eval')) if isinstance(n, ast.Name)}
        bad = [n for n in found if '__at_' in n or '__index' in n or '__prev' in n]
        own = [n for n in found if n in states and n != f.src and not copies and src.strats and mixing is None]
        if bad or own:
            why = f'uses {sc.strjoin(bad)} (a fixed or other level of an array)' if bad else f'uses {sc.strjoin(own)} at its own level ({own[0]}[i]): mixing within each level, which needs a contact matrix by level (ss.contacts) rather than the total'
            src.skip(f'{f.src or ""} -> {f.dst or ""}', why)
            src.flows.remove(f)
    for var in list(src.exprs):
        src.exprs[var] = re.sub(r'\b(\w+)__sum\b', r'\1', src.exprs[var]) # sum(I) is the number in I over every level, which is what I.n is in a stratified model
    for f in src.flows:
        if f.rate:
            f.rate = re.sub(r'\b(\w+)__sum\b', r'\1', f.rate)

    # Run settings
    cfg = dict(solver='binomial' if discrete else 'ode', unit=unit)
    if discrete: # odin's discrete-time models take whole steps
        cfg['substeps'] = 1
    if discrete and dt != 1:
        cfg['dt'] = dt
    if stop is not None:
        cfg['stop'] = getattr(ss, f'{unit}s')(stop) if unit else stop
    for word in ['interpolate(', 'delay(']:
        if word in all_text and not any(word in n for n in notes):
            src.skip(word.rstrip('('), f'{word.rstrip("(")}() is used and was not translated (an interpolated input could be an ss.timeseries)')
    if re.search(r'\b(t|time|step)\b', all_text):
        src.skip('time', 'an equation uses the time (t, time, or step); a value that changes in time could be an ss.timeseries')
    cfg.update(kwargs)
    src.notes += notes
    return build(src, verbose=verbose, **cfg)


def flow(src, frm, to, term, discrete, env, assign, dt, unit, notes):
    """ Add one flow from a term: in discrete time a draw, in continuous time a rate in total """
    text = ast.unparse(term)
    name = text if isinstance(term, ast.Name) else None

    # A delay of a flow into the same state: a fixed dwell time
    if name in src.exprs and src.exprs[name].startswith('delay('):
        call = ast.parse(src.exprs[name], mode='eval').body
        inner, tau = call.args[0], call.args[1]
        try:
            tau = float(evaluate(ast.unparse(tau), env))
        except KeyError:
            src.skip(name, f'a delay whose length "{ast.unparse(tau)}" is not a number')
            return
        num, den = factors(inner)
        flows_in = [n for n in num if isinstance(n, ast.Name) and n.id in src.exprs]
        rest = [n for n in num if n not in flows_in] + [ast.BinOp(ast.Constant(1), ast.Div(), d) for d in den]
        try:
            factor = float(np.prod([evaluate(ast.unparse(r), env) for r in rest])) if rest else 1.0
        except KeyError:
            factor = None
        if len(flows_in) == 1 and factor is not None:
            src.add_flow(frm, to, dur=ss.constant(getattr(ss, f'{unit}s')(tau)), label=f'delay({ast.unparse(inner)}, {tau:n})')
            notes.append(f'{name} = delay({ast.unparse(inner)}, {tau:n}) read as a fixed dwell time of {tau:n} in {frm}, whose outflows give the share who survive it ({factor:.6g})')
            src.delays.append((name, frm, tau, factor))
            src.skip(f'{name} (history)', f'for the first {tau:n}, odin\'s delay() gives the value at time 0 (as if the flow had been going on before the start), while the import starts with no one partway through {frm}; a workaround is an inflow to {to} for the first {tau:n} (ss.null >> {to} @ ss.timeseries(...))')
            return
        src.skip(name, f'a delay that is not a delay of one flow ({src.exprs[name]})')
        return

    # Discrete time: a draw
    if discrete:
        rhs = src.exprs.get(name)
        if rhs is None:
            src.skip(text, f'a term of an update() equation that is not a draw ({frm or ""} -> {to or ""})')
            return
        call = ast.parse(rhs, mode='eval').body
        fn = call.func.id if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) else None
        fn = draws.get(fn, fn) # Binomial() is rbinom()
        if fn == 'rbinom' and len(call.args) == 2:
            size, p = call.args
            if isinstance(size, ast.Name) and size.id == frm:
                rate = per_step(p, src.exprs, assign, dt)
                src.add_flow(frm, to, rate, name=name)
            else:
                src.skip(name, f'a binomial draw from "{ast.unparse(size)}", not from the state people leave ({frm}): a chained draw, which is a branch (p= on the transition)')
        elif fn == 'rpois' and len(call.args) == 1:
            src.add_flow(frm, to, ast.unparse(call.args[0]) if dt == 1 else f'({ast.unparse(call.args[0])}) / {dt}', per='total', name=name)
        else:
            src.skip(name, f'the draw "{rhs}" is not translated (only rbinom(state, p) or Binomial(), and rpois(rate) or Poisson(), are)')
        return

    # Continuous time: a rate in total, or an inflow
    src.add_flow(frm, to, text, per='total', name=name)
    return


def likelihood(src, var, rhs, data, env, used):
    """ An outcome for odin2's comparison to data, cases ~ Poisson(p * incidence), with its likelihood: returns why not, if it is not one """
    call = ast.parse(r2py(rhs), mode='eval').body
    fn = call.func.id if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) else None
    names = dict(Poisson=['lambda_'], NegativeBinomial=['size', 'mu'], Normal=['mean', 'sd']).get(fn)
    if var not in data:
        return f'{var} is not data (data())'
    elif names is None:
        return f'the likelihood {fn} is not translated (only Poisson, NegativeBinomial(size, mu), and Normal(mean, sd) are)'
    args = dict(zip(names, call.args)) | {k.arg:k.value for k in call.keywords}
    if set(args) != set(names):
        return f'the arguments are not {sc.strjoin(names)}'
    def par(node): # A parameter by name, or a number
        if isinstance(node, ast.Name) and node.id in env:
            used.add(node.id)
            return lambda ps: ps[node.id]
        try:
            value = float(evaluate(ast.unparse(node), env))
            return lambda ps: value
        except KeyError:
            return None
    mean = args.pop('lambda_' if fn == 'Poisson' else 'mu' if fn == 'NegativeBinomial' else 'mean')
    num, den = factors(mean)
    flows = [fl.name for fl in src.flows]
    known = [f for f in num if isinstance(f, ast.Name) and (f.id in src.outcomes or f.id in flows)]
    rest = [f for f in num if f not in known]
    if len(known) != 1 or den or len(rest) > 1 or (rest and not (isinstance(rest[0], ast.Name) and rest[0].id in env)):
        return f'its mean ({ast.unparse(mean)}) is not a count over a period or a flow, or a parameter times one'
    other = {k:par(v) for k,v in args.items()}
    if None in other.values():
        return f'{sc.strjoin(ast.unparse(v) for v in args.values())} is not a parameter or a number'
    if rest:
        used.add(rest[0].id)
    lik = dict(Poisson=lambda ps: ss.poisson(), NegativeBinomial=lambda ps: ss.nbinom(n=other['size'](ps)), Normal=lambda ps: ss.normal(scale=other['sd'](ps)))[fn]
    src.add_outcome(var, known[0].id, p=rest[0].id if rest else None, likelihood=lik)
    return None


def per_step(p, exprs, assign, dt):
    """ The rate that gives a probability p over a step: X from 1 - exp(-X) (or 1 - exp(-X * dt)), else -log(1 - p) """
    if isinstance(p, ast.Name) and p.id in exprs:
        p = ast.parse(exprs[p.id], mode='eval').body
    elif isinstance(p, ast.Name) and p.id in assign: # A probability of parameters only, from what it was written as (1 - exp(-gamma) is gamma)
        p = strip_index(ast.parse(r2py(assign[p.id][0][1]), mode='eval').body)
    per = lambda text: f'({text}) / {dt}' if dt != 1 else text
    if isinstance(p, ast.BinOp) and isinstance(p.op, ast.Sub) and isinstance(p.left, ast.Constant) and p.left.value == 1:
        e = p.right
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == 'exp' and len(e.args) == 1:
            x, minus = e.args[0], 0
            while isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.USub): # 1 - exp(-(X * dt))
                x, minus = x.operand, minus + 1
            num, den = factors(x)
            minus += sum(isinstance(f, ast.UnaryOp) and isinstance(f.op, ast.USub) for f in num + den)
            if minus % 2: # 1 - exp(-X), with the minus on any factor of X
                num, den = [strip_sign(f) for f in num], [strip_sign(f) for f in den]
                kept = [f for f in num if not (isinstance(f, ast.Name) and f.id == 'dt')]
                if len(kept) < len(num): # 1 - exp(-X * dt): the rate is X
                    return ast.unparse(product(kept, den))
                return per(ast.unparse(product(num, den)))
    return per(f'-log(1 - ({ast.unparse(p)}))')


def strip_sign(node):
    """ A factor without a unary minus, which signed_terms() has already counted """
    return node.operand if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) else node
