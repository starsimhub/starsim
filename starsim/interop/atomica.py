"""
Import an Atomica model (a framework workbook and a databook)

Atomica (https://atomica.tools) describes a model's structure in a framework workbook
(compartments, characteristics, parameters, and a transition matrix) and its values in a
databook. Compartments become states, characteristics results, and each parameter on the
transition matrix a transition; parameters with functions become expressions of the
counts. Atomica moves `value × dt` of a compartment each step for a rate or probability,
`dt/value` for a duration, and `value × dt` people for a number, which is the Euler
step, so the sim is stepped as Atomica steps. Several populations become a stratification
by population. A program book's programs become parameters of spending, whose coverage
changes the parameters they affect (see `read_progbook()`). Junctions, transfers, and
interactions are not translated.
"""
import ast
import re
import datetime
import numpy as np
import pandas as pd
import scipy.optimize as spo
import sciris as sc
import starsim as ss
from .core import Source, build

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
    types = {pop:get_type(pops, pop) for pop in pops.index}
    return types, out


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


def load_atomica(framework, databook=None, progbook=None, start=None, stop=None, dt=0.25, programs_start=None, verbose=None):
    """
    Read an Atomica model (a framework workbook and a databook) into a sim

    Args:
        framework (str): the framework workbook (.xlsx)
        databook (str): the databook (.xlsx); without one, the framework's default values are used
        progbook (str): the program book (.xlsx): each program becomes a parameter of spending (`spend_<program>`), whose coverage changes the parameters it affects
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
            src.skip(f'population type {t}', f'only the first population type ({types[0]}) is imported; {t} and its {len(parts)} compartments, characteristics, and parameters ({sc.strjoin(parts[:6])}{", ..." if len(parts) > 6 else ""}) would be a disease of their own')
        comps, chars, pars = mine(comps), mine(chars), mine(pars)

    # Transition matrix: sources in the first column, destinations along the top, in a block for each population type
    tm = sheets['Transitions'].dropna(how='all')
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
            try:
                src.exprs[p] = ast.unparse(fl.visit(ast.parse(text, mode='eval')))
            except SyntaxError: # e.g. a flow's rate, life_expectancy*:ddis
                src.skip(f'parameter {p}', f'its function ({func}) uses Atomica syntax that is not translated (":" for a flow\'s value), so neither are the transitions that use it')
                continue
            for fn in {n.func.id for n in ast.walk(ast.parse(src.exprs[p], mode='eval')) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)} & set(aggregations):
                src.skip(f'parameter {p}', f'its function ({func}) uses {fn}(), {aggregations[fn]}, which is not translated')
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
        return float(weight) > 0, {pop:(at_year(*entries[pop], start) if entries and pop in entries else None if default is None else float(default)) for pop in pops}
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
        if f == 'duration' and yes(get(pars, p, 'Timed')): # A timed duration: everyone leaves after it, a fixed dwell time
            vals = used.get(p, [])
            if vals and all(sc.isnumber(v) and v == vals[0] for v in vals):
                src.add_flow(frm, to, dur=ss.constant(ss.years(float(vals[0]))), name=tname, label=get(pars, p, 'Display Name'))
            else:
                src.skip(f'transition {tname}', f'"{p}" is a timed duration that is not one number ({vals}), which is not translated')
            continue
        rate = f'1 / {p}' if f == 'duration' else p
        per = 'total' if f == 'number' else 'person'
        src.add_flow(None if frm in sources else frm, to, rate, per=per, name=tname, label=get(pars, p, 'Display Name'))

    # The parameters that are used, by population
    for p,q in floors.items():
        vals = used.get(p, used.get(f'{p}_data', []))
        if not all(sc.isnumber(v) and float(v).is_integer() for v in vals):
            src.skip(f'parameter {q}', f'its function takes floor() of "{p}", which is not a whole number ({vals}); it was written as {p}')
    for p,vals in used.items():
        val = vals[0] if len(pops) == 1 or all(np.all(v == vals[0]) if not isinstance(v, ss.timeseries) else False for v in vals) else ss.by(population=vals)
        code = p.removesuffix('_data') if p not in pars.index else p # A parameter that programs affect keeps its databook value as <name>_data
        src.add_par(p, val, doc=get(pars, code, 'Display Name'))

    # Populations, and the initial numbers
    if len(pops) > 1:
        totals = init.sum(axis=1)
        src.stratify(population=list(pops), kind='space', split=list(totals/totals.sum())) # A place, so that a count in a rate (.n) is the count in each population, as in Atomica
        for j,s in enumerate(states[1:], start=1): # The first state takes the rest of each population
            vals = init[:, j]
            if vals.any():
                small = (vals > 0) & (vals <= 1) # Numbers of people that would be read as fractions, so all are given as the fraction of each population
                src.states[s].init = ss.by(population=list(vals/totals if small.any() else vals))
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
