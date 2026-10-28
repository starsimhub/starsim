"""
Import an Atomica model (a framework workbook and a databook)

Atomica (https://atomica.tools) describes a model's structure in a framework workbook
(compartments, characteristics, parameters, and a transition matrix) and its values in a
databook. Compartments become states, characteristics results, and each parameter on the
transition matrix a transition; parameters with functions become expressions of the
counts. Atomica moves `value × dt` of a compartment each step for a rate or probability,
`dt/value` for a duration, and `value × dt` people for a number, which is the Euler
step, so the sim is stepped as Atomica steps. Several populations become a stratification
by population, the databook's transfers between them (e.g. aging) transfers along it, and an
average or sum over the populations (`SRC_POP_AVG()`) a count in all of them, where everyone
interacts with everyone alike. A junction (which people pass through within a step) is not
a state: the transitions into it branch to where its outflows lead. A calibration (the
y-factors Atomica saves with `ParameterSet.save_calibration()`) scales the values it calibrates.
A program book's programs become parameters of spending, whose coverage changes the
parameters they affect (see `read_progbook()`).
"""
import ast
import re
import datetime
import numpy as np
import pandas as pd
import scipy.optimize as spo
import sciris as sc
import starsim as ss
from .core import Source, Code, build

__all__ = ['load_atomica']

yes = lambda v: str(v).strip().lower() in ['y', 'yes', 'true', '1']
aggregations = dict(SRC_POP_AVG='an average over the populations that interact with each one', SRC_POP_SUM='a sum over the populations that interact with each one', TGT_POP_AVG='an average over the populations each one interacts with', TGT_POP_SUM='a sum over the populations each one interacts with', POP_AVG='an average over the populations of a type', POP_SUM='a sum over the populations of a type') # Atomica's functions across populations


def blank(v):
    """ Whether a spreadsheet cell is empty """
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip())


def year(v):
    """ A column heading as a year, or None """
    if isinstance(v, (datetime.date, pd.Timestamp)):
        return float(v.year)
    if sc.isnumber(v) and not blank(v):
        return float(v)
    try:
        return float(str(v)[:4]) if str(v)[:4].isdigit() else None
    except ValueError:
        return None


def table(sheets, name):
    """ A framework sheet as a dataframe with its first row as the header, keyed by code name """
    df = {k.lower():v for k,v in sheets.items()}.get(name.lower()) # In any case ("Population types")
    if df is None:
        return pd.DataFrame()
    df = df.dropna(how='all')
    df.columns = [str(c).strip().title() for c in df.iloc[0]] # Atomica reads its headings in any case ("Display name" or "Display Name")
    df = df.iloc[1:]
    df = df[[not blank(v) for v in df.iloc[:, 0]]]
    return df.set_index(df.columns[0])


def read_databook(filename):
    """ The databook's populations, and each quantity's values by population: {name: {pop: (constant, {year: value})}} """
    sheets = pd.read_excel(filename, sheet_name=None, header=None)
    pops = table(sheets, 'Population Definitions')
    out = {}
    for sname,df in sheets.items():
        if sname in ['Population Definitions', 'Transfers', 'Interactions']: # Tables between populations: see read_between()
            continue
        rows = df.values.tolist()
        i = 0
        while i < len(rows):
            row = rows[i]
            if not blank(row[0]) and any(str(c).strip() in ['Provenance', 'Units'] for c in row[1:3]): # The heading of a quantity
                head = [str(c).strip() if not blank(c) else '' for c in row]
                const = head.index('Constant') if 'Constant' in head else None
                years = {j:year(row[j]) for j in range(1, len(row)) if year(row[j]) is not None and head[j] not in ['Constant', 'Uncertainty']}
                entries = {}
                i += 1
                while i < len(rows) and not blank(rows[i][0]):
                    r = rows[i]
                    c = r[const] if const is not None and not blank(r[const]) else None
                    entries[str(r[0]).strip()] = (c, {y:float(r[j]) for j,y in years.items() if not blank(r[j])})
                    i += 1
                out[str(row[0]).strip()] = entries
            else:
                i += 1
    types = {pop:get_type(pops, pop) for pop in pops.index}
    return types, out


def read_between(filename, sheet):
    """
    A databook's quantities between populations, its transfers or its interactions, as {code: {(from, to): (units, constant, {year: value})}}

    Each has a table of its code, name, and types, a matrix of which pairs of populations have a value, and then a row for each pair.
    """
    sheets = pd.read_excel(filename, sheet_name=None, header=None)
    df = next((v for k,v in sheets.items() if k.strip().lower() == sheet.lower()), None)
    out = {}
    if df is None:
        return out
    rows = df.values.tolist()
    code = head = None
    for i,r in enumerate(rows):
        first = '' if blank(r[0]) else str(r[0]).strip()
        if first == 'Abbreviation': # The heading of a quantity: its code, name, and types are in the next row
            code, head = str(rows[i+1][0]).strip(), None
            out[code] = {}
        elif first == 'From population': # The heading of its values
            head = [str(c).strip() if not blank(c) else '' for c in r]
        elif head is not None and str(r[1]).strip() == '--->':
            col = {name:head.index(name) for name in ['Units', 'Constant'] if name in head}
            const = r[col['Constant']] if 'Constant' in col and not blank(r[col['Constant']]) else None
            series = {year(h):float(r[j]) for j,h in enumerate(head) if j > col.get('Constant', 3) and year(h) is not None and sc.isnumber(r[j]) and not blank(r[j])}
            units = str(r[col['Units']]).strip() if 'Units' in col and not blank(r[col['Units']]) else None
            if const is not None or series:
                out[code][(first, str(r[2]).strip())] = (units, None if const is None else float(const), series)
    return out


def read_calibration(filename):
    """
    The y-factors of a calibration saved by Atomica (`ParameterSet.save_calibration()`): {code: (meta, {pop: factor})}, and for
    a transfer or an interaction, {(code, from pop): (meta, {to pop: factor})}
    """
    df = pd.read_excel(filename, sheet_name=0, header=None)
    head = [str(c).strip() for c in df.iloc[0]]
    out = {}
    for r in df.iloc[1:].values.tolist():
        if blank(r[0]):
            continue
        key = str(r[0]).strip() if blank(r[1]) else (str(r[0]).strip(), str(r[1]).strip())
        out[key] = (float(r[2]), {pop:float(v) for pop,v in zip(head[3:], r[3:]) if not blank(v)})
    return out


def read_progbook(filename):
    """
    The programs of a program book: who each targets, its spending and unit cost, and its effect on each parameter it affects

    Returns (programs, effects): programs as {name: dict(pops=, comps=, spend=, cost=, other=)}, with the names as written in the book;
    and effects as {parameter: {pop: dict(base=, interaction=, outcomes={program: value})}}, by the parameter's name as written there
    """
    sheets = pd.read_excel(filename, sheet_name=None, header=None)
    find = lambda name: next((df for k,df in sheets.items() if k.strip().lower() == name), None)

    # Targeting: Y or N for each population, then for each compartment, in two blocks of columns
    rows = find('program targeting').values.tolist()
    top, head = rows[0], rows[1]
    starts = [j for j,v in enumerate(top) if not blank(v)]
    pop_cols = range(starts[0], starts[1]) if len(starts) > 1 else range(starts[0], len(head))
    comp_cols = range(starts[1], len(head)) if len(starts) > 1 else range(0)
    programs = {}
    for row in rows[2:]:
        if blank(row[0]):
            continue
        name = str(row[1] if not blank(row[1]) else row[0]).strip()
        programs[name] = sc.objdict(abbr=str(row[0]).strip(), pops=[str(head[j]).strip() for j in pop_cols if yes(row[j])], comps=[str(head[j]).strip() for j in comp_cols if not blank(head[j]) and yes(row[j])], spend=None, cost=None, other=[])

    # Spending: a block for each program, with a row for each quantity, an assumption, and values by year
    rows = find('spending data').values.tolist()
    current = None
    for row in rows:
        label = str(row[0]).strip() if not blank(row[0]) else ''
        if not blank(row[1]) and str(row[1]).strip() == 'Units': # The heading of a program's block
            head = row
            current = next((p for p in programs.values() if label in [p.abbr] or label == next(k for k,v in programs.items() if v is p)), None)
            continue
        if current is None or not label:
            continue
        assume = next((row[j] for j,h in enumerate(head) if str(h).strip() == 'Assumption' and not blank(row[j])), None)
        series = {year(h):float(row[j]) for j,h in enumerate(head) if j > 0 and year(h) is not None and not blank(row[j])}
        value = (float(assume), {}) if assume is not None else (None, series) if series else None
        key = label.lower()
        if key.startswith('annual spend'):
            current.spend = value
        elif key.startswith('unit cost'):
            current.cost = value
        elif value is not None and not key.startswith('coverage'): # Capacity constraints and saturation
            current.other.append(label)

    # Effects: a block for each parameter, with a row for each population: its baseline, how coverage combines, and each program's outcome
    rows = find('program effects').values.tolist()
    effects = {}
    current = None
    for row in rows:
        if not blank(row[1]) and str(row[1]).strip() == 'Baseline value':
            head = row
            current = effects.setdefault(str(row[0]).strip(), {})
            continue
        if current is None or blank(row[0]):
            continue
        cols = {j:str(h).strip() for j,h in enumerate(head) if j >= 5 and not blank(h)}
        current[str(row[0]).strip()] = sc.objdict(base=None if blank(row[1]) else float(row[1]), interaction=None if blank(row[2]) else str(row[2]).strip(),
                                                   impact=None if blank(row[3]) else str(row[3]).strip(), outcomes={cols[j]:float(row[j]) for j in cols if not blank(row[j])})
    return programs, effects


def get_type(df, code):
    """ The population type of a row of a framework or databook table (None if it has none) """
    val = df.loc[code, 'Population Type'] if 'Population Type' in df.columns else None
    return None if blank(val) else str(val).strip()


def at_year(constant, series, t):
    """ A databook value at a time, as Atomica interpolates it """
    if constant is not None:
        return float(constant)
    if not series:
        return None
    ys = sorted(series)
    return float(np.interp(t, ys, [series[y] for y in ys]))


def as_value(constant, series, factor=1.0, lo=None, hi=None):
    """ A databook value as a parameter: a number, or a series that varies linearly in time; times a calibration's y-factor, and within the parameter's limits """
    fix = lambda v: float(np.clip(v*factor, -np.inf if lo is None else lo, np.inf if hi is None else hi))
    if constant is not None:
        return fix(constant)
    if len(series) == 1:
        return fix(list(series.values())[0])
    points = [(float(y), fix(v)) for y,v in sorted(series.items())]
    kept = [pt for i,pt in enumerate(points) if i in (0, len(points) - 1) or not points[i-1][1] == pt[1] == points[i+1][1]] # Not a point equal to those on either side, which changes nothing (e.g. a databook given month by month)
    if len(kept) == 2 and kept[0][1] == kept[1][1]:
        return kept[0][1]
    return ss.timeseries(dict(kept), mode='linear')


def same(a, b):
    """ Whether two parameter values are the same: numbers, or series that vary in time """
    if isinstance(a, ss.timeseries) or isinstance(b, ss.timeseries):
        return isinstance(a, ss.timeseries) and isinstance(b, ss.timeseries) and list(a.times) == list(b.times) and np.allclose(a.values, b.values, rtol=1e-12, atol=0)
    return np.all(a == b)


def atomica_expr(text):
    """ An Atomica function in Python syntax: ^ is a power """
    return str(text).strip().replace('^', '**')


class StepOf(ast.NodeTransformer):
    """ Atomica's dt in a function: the step """
    def __init__(self, dt):
        self.dt = dt
        return

    def visit_Name(self, node):
        return ast.Constant(self.dt) if node.id == 'dt' else node


def safe_div(text):
    """ Atomica's sdiv(a, b), a/b or 0 where b is 0, in a rate (where b is not negative: a count) """
    class Div(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if getattr(node.func, 'id', None) == 'sdiv' and len(node.args) == 2:
                a, b = [ast.unparse(arg) for arg in node.args]
                return ast.parse(f'({a}) * (({b}) > 0) / (({b}) + (({b}) <= 0))', mode='eval').body
            return node
    return text if 'sdiv' not in str(text) else ast.unparse(Div().visit(ast.parse(str(text), mode='eval')))


def load_atomica(framework, databook=None, progbook=None, calibration=None, start=None, stop=None, dt=0.25, programs_start=None, verbose=None):
    """
    Read an Atomica model (a framework workbook and a databook) into a sim

    Args:
        framework (str): the framework workbook (.xlsx)
        databook (str): the databook (.xlsx); without one, the framework's default values are used
        progbook (str): the program book (.xlsx): each program becomes a parameter of spending (`spend_<program>`), whose coverage changes the parameters it affects
        calibration (str): a calibration saved by Atomica (.xlsx, its y-factors): each quantity it calibrates is scaled by its y-factor in each population
        programs_start (float): the year the programs start to set the parameters they affect, as Atomica's `ProgramInstructions(start_year=)` (default: the start)
        start (float): the first year (default: the first year of data, as in Atomica)
        stop (float): the last year (default: as in Atomica, five years after the data, or 2035 if the model has births)
        dt (float): the step, in years (default: Atomica's 0.25)
        verbose (bool): whether to print how each transition was read

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated

    **Example**:

        import starsim.interop
        sim = ss.interop.load('sir_framework.xlsx', format='atomica', databook='sir_databook.xlsx', dt=1)
    """
    sheets = pd.read_excel(framework, sheet_name=None, header=None)
    comps = table(sheets, 'Compartments')
    chars = table(sheets, 'Characteristics')
    pars = table(sheets, 'Parameters')
    about = table(sheets, 'About')
    name = str(about.index[0]) if len(about) else sc.path(framework).stem
    src = Source(name=name, format='Atomica', file=str(framework))
    get = lambda df, key, col: df.loc[key, col] if col in df.columns and not blank(df.loc[key, col]) else None

    # Population types: the first is imported, and the others (with the interactions between them) are listed
    types = list(table(sheets, 'Population Types').index) or [None]
    if len(types) > 1:
        mine = lambda df: df[[get_type(df, k) in [None, types[0]] for k in df.index]]
        for t in types[1:]:
            parts = [k for df in [comps, chars, pars] for k in df.index if get_type(df, k) == t]
            if parts:
                    src.skip(f'population type {t}', f'only the first population type ({types[0]}) is imported; {t} and its {len(parts)} compartments, characteristics, and parameters ({sc.strjoin(parts[:6])}{", ..." if len(parts) > 6 else ""}) would be a disease of their own')
        comps, chars, pars = mine(comps), mine(chars), mine(pars)

    # Transition matrix: sources in the first column, destinations along the top, in a block for each population type
    tm = {k.lower():v for k,v in sheets.items()}['transitions'].dropna(how='all') # In any case, as table()
    dests = []
    links = []
    for row in tm.values.tolist():
        frm = str(row[0]).strip() if not blank(row[0]) else None
        if frm not in comps.index and (frm is None or frm in types or not dests): # The heading of a block
            dests = [str(c).strip() if not blank(c) else None for c in row[1:]]
            continue
        for to,cell in zip(dests, row[1:]):
            if not blank(cell) and frm in comps.index and to in comps.index:
                for par in str(cell).split(','):
                    links.append((frm, to, par.strip()))

    # Data
    pops, data = read_databook(databook) if databook is not None else ({'default':None}, {})
    pops = [pop for pop,t in pops.items() if len(types) == 1 or t in [None, types[0]]]
    def values(code, df):
        """ The databook entries for a code name, by population """
        label = get(df, code, 'Display Name') or code
        return data.get(label, data.get(code))
    years = sorted({y for entries in data.values() for c,s in entries.values() for y in s})
    sources = [c for c in comps.index if yes(get(comps, c, 'Is Source'))]
    if start is None:
        start = years[0] if years else 2000.0
    if stop is None:
        stop = 2035.0 if sources else (years[-1] + 5 if years else 2035.0)
    yfs = read_calibration(calibration) if calibration is not None else {}
    def factor(key, pop):
        """ A quantity's y-factor in a population, times its meta y-factor (1 if it is not calibrated) """
        meta, by_pop = yfs.get(key, (1.0, {}))
        return meta*by_pop.get(pop, 1.0)

    # States: every compartment that is not a source or a junction
    junctions = [c for c in comps.index if yes(get(comps, c, 'Is Junction'))]
    for c in comps.index:
        if c not in sources and c not in junctions:
            src.add_state(c, label=get(comps, c, 'Display Name'))
    states = list(src.states.keys())
    src.declared = True # Every compartment is people, including those of a cascade joined to the others only by numbers (from a source, into a sink)

    # Characteristics: sums of compartments (and other characteristics), maybe over a denominator
    def weights(code, seen=()):
        """ A characteristic or compartment as the weight of each compartment in it """
        if code in states:
            return {code:1.0}
        elif code in junctions: # Always empty
            return {}
        comp = get(chars, code, 'Components')
        out = {}
        for part in [p.strip() for p in str(comp).split(',') if p.strip()]:
            if part in seen:
                errormsg = f'The characteristic "{code}" refers to itself'
                raise ValueError(errormsg)
            for k,v in weights(part, seen + (code,)).items():
                out[k] = out.get(k, 0) + v
        return out
    def char_expr(code):
        num = ' + '.join(weights(code))
        den = get(chars, code, 'Denominator')
        return f'({num})' if den is None else f'({num}) / ({" + ".join(weights(den))})'
    for c in chars.index:
        src.exprs[c] = char_expr(c)
        src.results[c] = c

    # Parameters: values from the databook (else the framework's default), or a function; times a calibration's y-factors, and within their limits, as Atomica keeps them
    fmt = {p:str(get(pars, p, 'Format') or '').strip().lower() for p in pars.index}
    limits = {p:(get(pars, p, 'Minimum Value'), get(pars, p, 'Maximum Value')) for p in pars.index}
    used = {} # Parameters with values, by population
    floors = {} # Parameters inside floor(), written as themselves if they are whole numbers
    raw = {} # The functions as written, before their y-factors and limits
    aggs = {} # Functions over populations (SRC_POP_AVG(), ...), worked out once the values are known
    def bounded(expr, p):
        """ A function, times its y-factors, within its limits """
        scale = [factor(p, pop) for pop in pops]
        if any(f != 1 for f in scale):
            if all(f == scale[0] for f in scale):
                expr = f'{scale[0]} * ({expr})'
            else:
                src.add_par(f'{p}_yfactor', ss.by(population=scale), doc=f'The calibration\'s y-factor of {p}')
                expr = f'{p}_yfactor * ({expr})'
        lo, hi = limits[p]
        expr = expr if lo is None else f'max({expr}, {float(lo)})'
        return expr if hi is None else f'min({expr}, {float(hi)})'
    for p in pars.index:
        func = get(pars, p, 'Function')
        if func is not None:
            text = atomica_expr(func)
            if ':' in text: # The flow along a transition (a:b), out of a compartment (a:), into one (:b), or through a parameter's transitions (p:flow), which only an output can use
                src.skip(f'parameter {p}', f'an output: its function ({func}) uses flows (":"), which are not translated')
                continue
            node = ast.parse(text, mode='eval')
            if isinstance(node.body, ast.Call) and getattr(node.body.func, 'id', None) in aggregations:
                aggs[p] = node.body
                continue
            fl = FloorOfName()
            raw[p] = ast.unparse(StepOf(dt).visit(fl.visit(node)))
            src.exprs[p] = bounded(raw[p], p)
            for fn in {n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)} & set(aggregations):
                src.skip(f'parameter {p}', f'its function ({func}) uses {fn}() inside another function, which is not translated')
            for name in fl.names:
                floors[name] = p
            if fl.other:
                src.skip(f'parameter {p}', f'its function uses floor() of an expression ({text}), which is not translated')
            continue
        entries = values(p, pars)
        default = get(pars, p, 'Default Value')
        lo, hi = limits[p]
        vals = [as_value(*entries[pop], factor(p, pop), lo, hi) if entries and pop in entries else (as_value(float(default), {}, factor(p, pop), lo, hi) if default is not None else None) for pop in pops]
        if any(v is None for v in vals):
            src.skip(f'parameter {p}', 'has no value in the databook and no default')
            continue
        used[p] = vals

    # Functions over populations: where every pair of populations interacts alike, an average of x weighted by w (SRC_POP_AVG(x, interaction, w)), where x is a
    # count over w (a prevalence), is the count over w in all of them, and a sum (SRC_POP_SUM(x, interaction)) is the count in all of them
    between = read_between(databook, 'Interactions') if databook is not None else {}
    def expand(text, seen=()):
        """ An expression with its named expressions (parameters with functions, and characteristics) written out """
        node = ast.parse(text, mode='eval').body if isinstance(text, str) else text
        class Sub(ast.NodeTransformer):
            def visit_Name(self, n):
                if n.id in src.exprs:
                    if n.id in seen:
                        errormsg = f'The function of "{n.id}" refers to itself'
                        raise ValueError(errormsg)
                    return expand(src.exprs[n.id], seen + (n.id,))
                return n
        return Sub().visit(node)
    def uniform(name):
        """ Whether a parameter has the same value in every population """
        if name in used:
            return all(same(v, used[name][0]) for v in used[name])
        return name in src.pars and not isinstance(src.pars[name].value, ss.by)
    def stateless(node):
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        return not names & set(states) and all(uniform(n) for n in names - {'max', 'min', 'exp', 'log', 'sqrt', 'abs', 'floor', 'sdiv'})
    def totalize(node):
        """ An expression of counts in a population, as the same counts in every population: it adds counts, times coefficients that are the same in every population """
        if stateless(node):
            return ast.unparse(node)
        elif isinstance(node, ast.Name) and node.id in states:
            return f'everywhere({node.id})'
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return f'({totalize(node.left)} {"+" if isinstance(node.op, ast.Add) else "-"} {totalize(node.right)})'
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and (stateless(node.left) or stateless(node.right)):
            return f'{totalize(node.left)} * {totalize(node.right)}'
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and stateless(node.right):
            return f'{totalize(node.left)} / ({ast.unparse(node.right)})'
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return f'-{totalize(node.operand)}'
        errormsg = f'{ast.unparse(node)} is not a sum of counts with the same coefficients in every population'
        raise ValueError(errormsg)
    def across(fn, args):
        """ A function over populations as an expression of counts in all of them """
        x, inter, w = (args + [None, None])[:3]
        if fn not in ['SRC_POP_AVG', 'SRC_POP_SUM', 'TGT_POP_AVG', 'TGT_POP_SUM'] or inter is None:
            raise ValueError('only an average or sum through an interaction is')
        if x not in src.exprs and x not in used:
            raise ValueError(f'{x} is not translated')
        ws = [as_value(c, series, factor((inter, a), b)) for (a,b),(units,c,series) in between.get(inter, {}).items() if a in pops and b in pops]
        if len(ws) < len(pops)**2 or not all(same(v, ws[0]) for v in ws):
            raise ValueError(f'the interaction {inter} is not the same between every pair of populations')
        if fn.endswith('AVG'): # The weight of the interaction cancels
            if w is None:
                raise ValueError('an average with no weighting is not')
            node = ast.parse(raw.get(x, x), mode='eval').body
            over = isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'sdiv' or isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
            num, den = (node.args if isinstance(node, ast.Call) else [node.left, node.right]) if over else (None, None)
            if num is None or ast.unparse(expand(den)) != ast.unparse(expand(w)):
                raise ValueError(f'{x} is not a count over {w}')
            src.notes.append(f'{fn}({", ".join(args)}): the limits of {x} are taken not to bind')
            return f'{totalize(expand(num))} / ({totalize(expand(w))})'
        if w is not None:
            raise ValueError('a sum with a weighting is not')
        weight = ws[0]
        if isinstance(weight, ss.timeseries):
            src.add_par(f'{inter}_weight', weight, doc=f'The interaction {inter}, the same between every pair of populations')
            weight = f'{inter}_weight'
        return totalize(expand(x)) if weight == 1 else f'{weight} * {totalize(expand(x))}'
    for p,call in aggs.items():
        fn = call.func.id
        try:
            src.exprs[p] = bounded(across(fn, [ast.unparse(a) for a in call.args]), p)
        except ValueError as E:
            src.skip(f'parameter {p}', f'its function ({ast.unparse(call)}) is {aggregations[fn]}, which is not translated: {E}')

    # Programs: spending buys a number of people reached a year, coverage is that number over everyone it targets (at most all of them), and each
    # parameter a program affects moves from its baseline towards the program's outcome in proportion to its coverage (Atomica's additive interaction)
    programs = sc.objdict() # The spending parameter of each program, by its name in the book
    if progbook is not None:
        progs, effects = read_progbook(progbook)
        pop_names = pop_labels(databook) if databook is not None else {}
        comp_names = {str(get(comps, c, 'Display Name') or c):c for c in comps.index}
        par_names = {str(get(pars, p, 'Display Name') or p):p for p in pars.index}
        slugs = {}
        for pname,prog in progs.items():
            slug = re.sub(r'\W+', '_', prog.abbr).strip('_').lower()
            reach = [comp_names.get(c, c) for c in prog.comps]
            targets = [pop_names.get(q, q) for q in prog.pops]
            if prog.spend is None or prog.cost is None:
                src.skip(f'program {pname}', 'it has no spending or no unit cost in the program book')
                continue
            if sorted(targets) != sorted(pops) and len(pops) > 1:
                src.skip(f'program {pname}', f'it targets some populations only ({sc.strjoin(targets)}), whose coverage is not translated')
                continue
            if prog.other:
                src.skip(f'program {pname}', f'its {sc.strjoin(prog.other)} are not translated')
            src.add_par(f'spend_{slug}', as_value(*prog.spend), doc=f'Spending on {pname} ($ a year)')
            src.add_par(f'cost_{slug}', as_value(*prog.cost), doc=f'Unit cost of {pname} ($ a person)')
            src.exprs[f'cov_{slug}'] = f'min(spend_{slug} / cost_{slug} / everywhere({" + ".join(reach)}), 1)'
            programs[pname] = f'spend_{slug}'
            slugs[pname] = slug
        if slugs:
            src.notes.append('programs: where the coverages of the programs that affect a parameter add up to more than 1, they are scaled down to add up to 1 (Atomica splits the excess between additive and random overlap)')
        src.add_par('programs_on', ss.timeseries({float(start): 0.0, float(programs_start if programs_start is not None else start): 1.0}), doc='Whether the programs set the parameters they affect')
        for label,rows in effects.items():
            p = par_names.get(label, label if label in pars.index else None)
            if p is None:
                src.skip(f'program effects on {label}', 'the parameter is not in the framework')
                continue
            if any(r.interaction not in [None, 'Additive'] or r.impact is not None for r in rows.values()):
                src.skip(f'program effects on {p}', 'their coverage or impact interaction is not additive, which is not translated')
                continue
            by_pop = [rows.get(next((k for k,v in pop_names.items() if v == pop), pop), rows.get(pop)) for pop in pops]
            if any(r is None for r in by_pop):
                src.skip(f'program effects on {p}', 'some populations have no row')
                continue
            value = lambda vals: vals[0] if len(pops) == 1 else ss.by(population=list(vals))
            base = [r.base or 0.0 for r in by_pop]
            src.add_par(f'{p}_programs', value(base), doc=f'{label}, under the programs, with none of them')
            terms, covs = [], []
            for pname,slug in slugs.items():
                diffs = [r.outcomes[pname] - b if pname in r.outcomes else 0.0 for r,b in zip(by_pop, base)]
                if any(diffs):
                    src.add_par(f'{p}_{slug}', value(diffs), doc=f'The effect of full coverage of {pname} on {label}')
                    terms.append(f'cov_{slug} * {p}_{slug}')
                    covs.append(f'cov_{slug}')
            under = f'({" + ".join(terms)}) / max({" + ".join(covs)}, 1) + {p}_programs' if terms else f'{p}_programs' # Coverages that add up to more than everyone are scaled down to everyone
            lo, hi = get(pars, p, 'Minimum Value'), get(pars, p, 'Maximum Value')
            under = under if lo is None else f'max({under}, {float(lo)})' # Within the parameter's limits, as Atomica keeps it
            under = under if hi is None else f'min({under}, {float(hi)})'
            if p in src.exprs: # A parameter with a function: the function without the programs
                before = f'({src.exprs[p]})'
            elif p in used:
                used[f'{p}_data'] = used.pop(p)
                before = f'{p}_data'
            else:
                src.skip(f'program effects on {p}', 'the parameter has no value to start from')
                continue
            src.exprs[p] = f'(1 - programs_on) * {before} + programs_on * ({under})'

    # Initial numbers, solved as Atomica does: from the compartments and characteristics with a setup weight (by default, those in the databook or with a default), by least squares with the smallest norm
    def setup(code, df):
        """ Whether a compartment or characteristic is used to find the initial numbers, and its value for each population """
        weight = get(df, code, 'Setup Weight')
        if weight is None:
            weight = float(get(df, code, 'Databook Page') is not None or get(df, code, 'Default Value') is not None)
        entries = values(code, df)
        default = get(df, code, 'Default Value')
        return float(weight) > 0, {pop:(at_year(*entries[pop], start)*factor(code, pop) if entries and pop in entries else None if default is None else float(default)*factor(code, pop)) for pop in pops}
    sinks = [c for c in comps.index if yes(get(comps, c, 'Is Sink'))]
    init = np.zeros((len(pops), len(states)))
    for i,pop in enumerate(pops):
        targets, zero = [], set()
        for code,df in [(c, comps) for c in states if c not in sinks] + [(c, chars) for c in chars.index]:
            use, vals = setup(code, df)
            val = vals[pop]
            if not use or val is None:
                continue
            if val == 0: # Everyone in it starts at zero, and is not solved for
                zero |= set(weights(code))
                continue
            den = get(chars, code, 'Denominator') if code in chars.index else None
            if den is not None: # A fraction of its denominator's own value in the databook
                dval = setup(den, chars if den in chars.index else comps)[1][pop]
                if dval is None:
                    src.skip(f'initial numbers ({pop})', f'the characteristic {code} is a fraction of {den}, which has no value in the databook')
                    continue
                val *= dval
            targets.append((weights(code), val))
        free = [s for s in states if s not in sinks and s not in zero]
        if targets and free:
            A = np.array([[w.get(s, 0) for s in free] for w,_ in targets])
            x = np.linalg.lstsq(A, np.array([v for _,v in targets]), rcond=None)[0]
            x = np.where(abs(x) < 1e-9*max(abs(x).max(), 1), 0.0, x) # Rounding, e.g. -2.8e-17, is no one
            init[i, [states.index(s) for s in free]] = x
            if np.any(x < -1e-3):
                src.skip(f'initial numbers ({pop})', f'the databook gives negative initial numbers ({sc.strjoin(f"{s}={v:g}" for s,v in zip(free, x) if v < -1e-3)}), which Atomica refuses (BadInitialization)')

    # Junctions: people pass through one within a step, so a transition into it branches to where its outflows lead, each with the share Atomica gives it
    # (its proportion over all of them; for a junction with a residual outflow ">", over at least 1, with the rest to the residual)
    outs = {j:[(to, p) for frm,to,p in links if frm == j] for j in junctions}
    def leaves(j, seen=()):
        """ The states people entering a junction end up in, through any junctions after it, and the share of them that goes to each, as an expression """
        props = [p for to,p in outs[j] if p != '>']
        total = ' + '.join(props)
        residual = len(props) < len(outs[j])
        found = {}
        for to,p in outs[j]:
            share = f'1 - min({total}, 1)' if p == '>' and props else '1' if p == '>' or len(outs[j]) == 1 else f'{p} / max({total}, 1)' if residual else f'{p} / ({total})'
            if to in junctions:
                if to in seen:
                    errormsg = f'The junctions {sc.strjoin(seen + (j, to))} lead back to each other, which Atomica does not allow'
                    raise ValueError(errormsg)
                parts = leaves(to, seen + (j,)).items()
            else:
                parts = [(to, '1')]
            for leaf,s in parts:
                found.setdefault(leaf, []).append(share if s == '1' else s if share == '1' else f'({share}) * ({s})')
        return {leaf:shares[0] if len(shares) == 1 else ' + '.join(f'({s})' for s in shares) for leaf,shares in found.items()}

    # Transitions
    def fixed(text):
        """ Whether an expression (with its named expressions written out) has the same value throughout: no states, and no parameters that vary in time """
        names = {n.id for n in ast.walk(expand(text)) if isinstance(n, ast.Name)} - {'max', 'min', 'exp', 'log', 'sqrt', 'abs', 'floor', 'sdiv'}
        varies = lambda v: isinstance(v, ss.parameters.TimeValue) or (isinstance(v, ss.by) and v.varies)
        return all(name in used and not any(varies(v) for v in used[name]) for name in names)
    for frm,to,p in links:
        if frm in junctions: # A branch of the transitions into it
            continue
        f = fmt.get(p, '')
        multi = sum(1 for a,b,q in links if a == frm and b == to) > 1
        tname = f'{frm}_to_{to}' + (f'_{p}' if multi else '')
        if f == 'proportion':
            src.skip(f'transition {tname}', f'"{p}" is a proportion, which only a junction uses')
            continue
        dst, shares = to, None
        if to in junctions:
            ends = leaves(to)
            dst = list(ends) if len(ends) > 1 else list(ends)[0]
            shares = list(ends.values())[:-1] + [None] if len(ends) > 1 else None
        if f == 'duration' and yes(get(pars, p, 'Timed')): # A timed duration: everyone leaves after it, a fixed dwell time
            vals = used.get(p, [])
            if vals and all(sc.isnumber(v) and v == vals[0] for v in vals) and to not in junctions:
                src.add_flow(frm, to, dur=ss.constant(ss.years(float(vals[0]))), name=tname, label=get(pars, p, 'Display Name'))
            else:
                src.skip(f'transition {tname}', f'"{p}" is a timed duration that is not one number ({vals}), which is not translated')
            continue
        rate = f'1 / {p}' if f == 'duration' else p
        per = 'total' if f == 'number' else 'person'
        froms = [a for a,b,q in links if q == p]
        if f == 'number' and frm not in sources and len(froms) > 1: # A number shared over every compartment it moves people out of (counted once for each of its transitions), as Atomica shares it
            rate, per = f'{p} / ({" + ".join(froms)})', 'person'
        if shares is not None and not all(fixed(share) for share in ends.values()): # Shares that change (with the numbers in states, or in time) are factors of the rates of a transition to each destination, which is the same flow, since people pass through a junction at once
            for leaf,share in ends.items():
                src.add_flow(None if frm in sources else frm, leaf, f'({rate}) * ({share})', per=per, name=f'{tname}_{leaf}', label=get(pars, p, 'Display Name'))
            continue
        src.add_flow(None if frm in sources else frm, dst, rate, per=per, name=tname, label=get(pars, p, 'Display Name'), p=shares)

    # The parameters that are used, by population
    for p,q in floors.items():
        vals = used.get(p, used.get(f'{p}_data', []))
        if not all(sc.isnumber(v) and float(v).is_integer() for v in vals):
            src.skip(f'parameter {q}', f'its function takes floor() of "{p}", which is not a whole number ({vals}); it was written as {p}')
    for p,vals in used.items():
        val = vals[0] if len(pops) == 1 or all(np.all(v == vals[0]) if not isinstance(v, ss.timeseries) else False for v in vals) else ss.by(population=vals)
        code = p.removesuffix('_data') if p not in pars.index else p # A parameter that programs affect keeps its databook value as <name>_data
        src.add_par(p, val, doc=get(pars, code, 'Display Name'))

    # Transfers between populations (e.g. aging): everyone in a population but the sinks moves to the other, at a rate (times its y-factors)
    transfers = {}
    for code,pairs in (read_between(databook, 'Transfers') if databook is not None and len(pops) > 1 else {}).items():
        for (a,b),(units,c,series) in pairs.items():
            kind = str(units).split()[0].lower()
            tname = re.sub(r'\W+', '_', f'{code}_{a}_to_{b}'.replace('+', 'pos').replace('- ', 'neg ')).strip('_').lower() # e.g. 15-18 HIV+ and 15-18 HIV-
            tname = next(f'{tname}_{i}' if i else tname for i in range(len(pairs) + 1) if (f'{tname}_{i}' if i else tname) not in transfers)
            if a not in pops or b not in pops:
                continue
            elif kind not in ['rate', 'probability', 'duration']:
                src.skip(f'transfer {tname}', f'it is a {units}, which is not translated')
                continue
            src.add_par(tname, as_value(c, series, factor((code, a), b), 0.0), doc=f'{code}, from {a} to {b} ({units})')
            rate = f'1/{tname}' if kind == 'duration' else tname
            transfers[tname] = Code(f'ss.transfer(population=({a!r}, {b!r}), rate={rate}, states={[s for s in states if s not in sinks]!r})')

    # Atomica's safe division, written out once nothing more needs to see it
    for k in src.exprs.keys():
        src.exprs[k] = safe_div(src.exprs[k])
    for flow in src.flows:
        flow.rate = flow.rate if flow.rate is None else safe_div(flow.rate)
        flow.p = flow.p if flow.p is None else [v if v is None else safe_div(v) for v in flow.p]

    # Populations, and the initial numbers
    if len(pops) > 1:
        totals = init.sum(axis=1)
        src.stratify(population=list(pops), kind='space', split=list(totals/totals.sum()), **transfers) # A place, so that a count in a rate (.n) is the count in each population, as in Atomica
        for j,s in enumerate(states): # Every state, as the fraction of each population
            vals = init[:, j]
            if vals.any():
                src.states[s].init = ss.by(population=list(vals/totals))
    else:
        for j,s in enumerate(states[1:], start=1):
            src.states[s].init = init[0, j]
    src.config.update(start=float(start), stop=float(stop), dt=float(dt), unit='year', solver='ode', integrator='euler', substeps=1, total_pop=float(init.sum()))
    sim = build(src, verbose=verbose)
    if programs:
        sim.imported.programs = programs # For ss.fit.optimize()
    return sim


def pop_labels(databook):
    """ The populations of a databook, as {full name: code} """
    df = table(pd.read_excel(databook, sheet_name=None, header=None), 'Population Definitions')
    col = next((c for c in df.columns if c in ['Full Name', 'Display Name']), None)
    return {str(df.loc[k, col]).strip():k for k in df.index} if col else {}


class FloorOfName(ast.NodeTransformer):
    """ floor(x) of a parameter x is written as x, which is right if x is a whole number (as Atomica's numbers of contacts usually are) """
    def __init__(self):
        self.names = [] # The parameters inside floor()
        self.other = False # Whether floor() was taken of anything else
        return

    def visit_Call(self, node):
        self.generic_visit(node)
        if getattr(node.func, 'id', None) == 'floor':
            if len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                self.names.append(node.args[0].id)
                return node.args[0]
            self.other = True
        return node
