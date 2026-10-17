"""
Import an Atomica model (a framework workbook and a databook)

Atomica (https://atomica.tools) describes a model's structure in a framework workbook
(compartments, characteristics, parameters, and a transition matrix) and its values in a
databook. Compartments become states, characteristics results, and each parameter on the
transition matrix a transition; parameters with functions become expressions of the
counts. Atomica moves `value × dt` of a compartment each step for a rate or probability,
`dt/value` for a duration, and `value × dt` people for a number, which is the Euler
step, so the sim is stepped as Atomica steps. Several populations become a stratification
by population. Programs, junctions, transfers, and interactions are not translated.
"""
import ast
import datetime
import numpy as np
import pandas as pd
import scipy.optimize as spo
import sciris as sc
import starsim as ss
from .core import Source, build

__all__ = ['load_atomica']

yes = lambda v: str(v).strip().lower() in ['y', 'yes', 'true', '1']


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
    df = sheets.get(name)
    if df is None:
        return pd.DataFrame()
    df = df.dropna(how='all')
    df.columns = [str(c).strip() for c in df.iloc[0]]
    df = df.iloc[1:]
    df = df[[not blank(v) for v in df.iloc[:, 0]]]
    return df.set_index(df.columns[0])


def read_databook(filename):
    """ The databook's populations, and each quantity's values by population: {name: {pop: (constant, {year: value})}} """
    sheets = pd.read_excel(filename, sheet_name=None, header=None)
    pops = table(sheets, 'Population Definitions')
    out = {}
    for sname,df in sheets.items():
        if sname == 'Population Definitions':
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
    return list(pops.index), out


def at_year(constant, series, t):
    """ A databook value at a time, as Atomica interpolates it """
    if constant is not None:
        return float(constant)
    if not series:
        return None
    ys = sorted(series)
    return float(np.interp(t, ys, [series[y] for y in ys]))


def as_value(constant, series):
    """ A databook value as a parameter: a number, or a series that varies linearly in time """
    if constant is not None:
        return float(constant)
    if len(series) == 1:
        return list(series.values())[0]
    return ss.timeseries({float(y):v for y,v in sorted(series.items())}, mode='linear')


def atomica_expr(text):
    """ An Atomica function in Python syntax: ^ is a power """
    return str(text).strip().replace('^', '**')


def load_atomica(framework, databook=None, progbook=None, start=None, stop=None, dt=0.25, verbose=None):
    """
    Read an Atomica model (a framework workbook and a databook) into a sim

    Args:
        framework (str): the framework workbook (.xlsx)
        databook (str): the databook (.xlsx); without one, the framework's default values are used
        progbook (str): the program book (.xlsx), whose programs are listed as not translated
        start (float): the first year (default: the first year of data, as in Atomica)
        stop (float): the last year (default: as in Atomica, five years after the data, or 2035 if the model has births)
        dt (float): the step, in years (default: Atomica's 0.25)
        verbose (bool): whether to print how each transition was read

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated

    **Example**:

        sim = ss.load('sir_framework.xlsx', format='atomica', databook='sir_databook.xlsx', dt=1)
    """
    sheets = pd.read_excel(framework, sheet_name=None, header=None)
    comps = table(sheets, 'Compartments')
    chars = table(sheets, 'Characteristics')
    pars = table(sheets, 'Parameters')
    about = table(sheets, 'About')
    name = str(about.index[0]) if len(about) else sc.path(framework).stem
    src = Source(name=name, format='Atomica', file=str(framework))
    get = lambda df, key, col: df.loc[key, col] if col in df.columns and not blank(df.loc[key, col]) else None

    # Transition matrix: sources in the first column, destinations along the top
    tm = sheets['Transitions'].dropna(how='all')
    dests = [str(c).strip() for c in tm.iloc[0, 1:]]
    links = []
    for row in tm.iloc[1:].values.tolist():
        frm = str(row[0]).strip()
        for to,cell in zip(dests, row[1:]):
            if not blank(cell):
                for par in str(cell).split(','):
                    links.append((frm, to, par.strip()))

    # Data
    pops, data = read_databook(databook) if databook is not None else (['default'], {})
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
    if progbook is not None:
        progs = pd.read_excel(progbook, sheet_name=None, header=None)
        src.skip('programs', f'the programs in {sc.path(progbook).name} (sheets {sc.strjoin(progs)}) are not translated: each is spending that sets the coverage of a delivery, which changes the parameters it affects (an ss.par of the coverage, as in the hand translation)')
    for key in ['Interactions', 'Transfers', 'Plots']:
        if key in sheets:
            src.skip(key, f'the "{key}" sheet is not translated')

    # States: every compartment that is not a source; a junction is not translated
    junctions = [c for c in comps.index if yes(get(comps, c, 'Is Junction'))]
    for c in comps.index:
        if c in sources:
            continue
        if c in junctions:
            src.skip(f'junction {c}', 'a junction (people pass through it within a step) is not translated')
        src.add_state(c, label=get(comps, c, 'Display Name'))
    states = list(src.states.keys())

    # Characteristics: sums of compartments (and other characteristics), maybe over a denominator
    def weights(code, seen=()):
        """ A characteristic or compartment as the weight of each compartment in it """
        if code in states:
            return {code:1.0}
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

    # Parameters: values from the databook (else the framework's default), or a function
    fmt = {p:str(get(pars, p, 'Format') or '').strip().lower() for p in pars.index}
    used = {} # Parameters with values, by population
    floors = {} # Parameters inside floor(), written as themselves if they are whole numbers
    for p in pars.index:
        func = get(pars, p, 'Function')
        if func is not None:
            text = atomica_expr(func)
            fl = FloorOfName()
            src.exprs[p] = ast.unparse(fl.visit(ast.parse(text, mode='eval')))
            for name in fl.names:
                floors[name] = p
            if fl.other:
                src.skip(f'parameter {p}', f'its function uses floor() of an expression ({text}), which is not translated')
            continue
        entries = values(p, pars)
        default = get(pars, p, 'Default Value')
        vals = [as_value(*entries[pop]) if entries and pop in entries else (float(default) if default is not None else None) for pop in pops]
        if any(v is None for v in vals):
            src.skip(f'parameter {p}', 'has no value in the databook and no default')
            continue
        used[p] = vals

    # Initial numbers, solved as Atomica does from the compartments and characteristics in the databook
    init = np.zeros((len(pops), len(states)))
    for i,pop in enumerate(pops):
        rows, rhs = [], []
        for code,df in [(c, comps) for c in states] + [(c, chars) for c in chars.index]:
            entries = values(code, df)
            if not entries or pop not in entries:
                continue
            val = at_year(*entries[pop], start)
            if val is None:
                continue
            row = np.zeros(len(states))
            for k,w in weights(code).items():
                row[states.index(k)] += w
            den = get(chars, code, 'Denominator') if code in chars.index else None
            if den is not None: # A fraction: the numerator minus the value times the denominator is zero
                for k,w in weights(den).items():
                    row[states.index(k)] -= val*w
                val = 0.0
            rows.append(row)
            rhs.append(val)
        if rows:
            A = np.array(rows)
            init[i] = spo.nnls(A, np.array(rhs))[0]
            if np.linalg.matrix_rank(A) < len([s for s in states if s not in [c for c in comps.index if yes(get(comps, c, 'Is Sink'))]]):
                src.skip(f'initial numbers ({pop})', 'the databook does not determine every compartment; the ones it leaves free were set by nonnegative least squares, where Atomica uses the setup weights')

    # Transitions
    for frm,to,p in links:
        if frm in junctions or to in junctions:
            continue
        f = fmt.get(p, '')
        multi = sum(1 for a,b,q in links if a == frm and b == to) > 1
        tname = f'{frm}_to_{to}' + (f'_{p}' if multi else '')
        if f == 'proportion':
            src.skip(f'transition {tname}', f'"{p}" is a proportion, which only a junction uses')
            continue
        rate = f'1 / {p}' if f == 'duration' else p
        per = 'total' if f == 'number' else 'person'
        src.add_flow(None if frm in sources else frm, to, rate, per=per, name=tname, label=get(pars, p, 'Display Name'))

    # The parameters that are used, by population
    for p,q in floors.items():
        vals = used.get(p, [])
        if not all(sc.isnumber(v) and float(v).is_integer() for v in vals):
            src.skip(f'parameter {q}', f'its function takes floor() of "{p}", which is not a whole number ({vals}); it was written as {p}')
    for p,vals in used.items():
        val = vals[0] if len(pops) == 1 or all(np.all(v == vals[0]) if not isinstance(v, ss.timeseries) else False for v in vals) else ss.by(population=vals)
        src.add_par(p, val, doc=get(pars, p, 'Display Name'))

    # Populations, and the initial numbers
    if len(pops) > 1:
        totals = init.sum(axis=1)
        src.stratify(population=list(pops), kind='space', split=list(totals/totals.sum())) # A place, so that a count in a rate (.n) is the count in each population, as in Atomica
        for j,s in enumerate(states[1:], start=1): # The first state takes the rest of each population
            if init[:, j].any():
                src.states[s].init = ss.by(population=list(init[:, j]))
    else:
        for j,s in enumerate(states[1:], start=1):
            src.states[s].init = init[0, j]
    src.config.update(start=float(start), stop=float(stop), dt=float(dt), unit='year', solver='ode', integrator='euler', substeps=1, total_pop=float(init.sum()))
    return build(src, verbose=verbose)


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
