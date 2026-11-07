"""
Export a sim to other formats

Exports use the compartmental form of declared diseases, turning a sim run as agents into it. `to_hub()` reads results alone, however they were made.

`ss.export_to(sim, format, filename=...)` writes a sim's declared diseases in another
format; see `docs/design/16-interop.md` in the lingua_franca repo. Each is read from the built model (each
disease's graph and engine). The formats carry less than a sim, and what is left out or
changed is listed as `ss.errors.LossyExport`. An export does not fail because something
cannot be carried (except MEmilio's, whose models are fixed: an SIR or SEIR, or an error).

- `sbml`: the compartmental form as species and reactions (https://sbml.org), for any SBML tool
- `odin`: the compartmental form as `deriv()` equations (https://mrc-ide.github.io/odin)

The other formats have importers too, and each exporter is beside its importer, as its
reverse: `epimodel` and `siminf` (R scripts), `summer2`, `epydemix`, `epipack`,
`epymorph` and `memilio` (Python scripts that build the model with the framework),
`camdl` and `emulsion` (their model files), `flepimop` (a config and the data files it
names), and `atomica` (a framework workbook, as bytes). The helpers they share are here: `in_unit()`
(the compartmental form in another unit of time), `flow_form()` (each flow as one of
the usual kinds, for formats that have kinds of flow), `timing()`, `declared_pars()`, and `counted()`.

The runs of a MultiSim are written as a submission to a hub (by scenario, as quantiles or samples) by `ss.converters.to_hub()`.

All are written from the compartmental form of the model (what
`sir.equations()` prints), with the numbers in the sim's unit of time: a dwell time
that is not exponential is its Erlang stages, each stratification is summed into one
group, a network is its mean contacts, and a parameter that varies in time is its formula
in the time (where the format can carry it). Each is read back by `ss.import_from()`, except a rate that varies in
time, which the importers list as not translated.

Used with: agents and compartments; declared and coded modules
"""
import re
import ast
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from .core import terms, factors, product, names
from .epimodel import to_epimodel
from .siminf import to_siminf
from .summer2 import to_summer2
from .epydemix import to_epydemix
from .epipack import to_epipack
from .epymorph import to_epymorph
from .camdl import to_camdl
from .emulsion import to_emulsion
from .flepimop import to_flepimop
from .memilio import to_memilio
from .atomica import to_atomica
from .starsim_json import to_json

__all__ = ['exporters', 'extensions', 'export_to', 'flatten', 'in_unit', 'flow_form', 'timing', 'declared_pars', 'counted', 'to_sbml', 'to_odin', 'to_hub']

math_funcs = dict(exp='exp', log='ln', sqrt='root', abs='abs', absolute='abs', sin='sin', cos='cos', floor='floor', ceiling='ceiling')
odin_funcs = dict(absolute='abs', minimum='min', maximum='max')
extrema = dict(minimum='lt', min='lt', maximum='gt', max='gt') # SBML Level 3 Version 1 has no <min> or <max>, so these are written as piecewise


#%% The compartmental form

def initialized(sim):
    """ The sim initialized and not yet run (so its state is the initial one): itself if it is, else a new one (see `sim.new()`) """
    if sim.initialized and not (sim.complete or sim.t.ti > 0):
        return sim
    return sim.new(verbose=0).init()


def compartmental(sim):
    """ An initialized sim whose diseases run as compartments, and what was changed to get it """
    sim = initialized(sim)
    if sim.solver != 'abm':
        return sim, None
    return sim.new(solver='ode', verbose=0).init(), 'changed: the agent-based sim was written as its compartmental form (an ODE)'


def strata(sim, mod):
    """ The stratifications of a disease, as text (e.g. 'age: 0-4, 5-17, 18+') """
    seen = {}
    for strat in list(getattr(getattr(sim, 'groups', None), 'strats', [])) + list(getattr(mod.graph, 'strats', [])):
        seen[strat.name] = f'{strat.name}: {", ".join(str(l) for l in strat.labels)}'
    return list(seen.values())


def rename(text, names, unit):
    """
    An expression with its names changed, as plain numbers and counts

    A typed value (`peryear(0.02)`) is its number in the sim's unit, a count (`(S + I).n`)
    is the sum it counts, a fraction (`I.frac`) that over `N`, and `mean(x)` (the mean
    of a dwell time) is `x`, whose value is the mean.
    """
    node = ast.parse(text, mode='eval').body
    class Sub(ast.NodeTransformer):
        def visit_Call(self, n):
            func = n.func.id if isinstance(n.func, ast.Name) else None
            if func == 'mean' and len(n.args) == 1:
                return self.visit(n.args[0])
            cls = getattr(ss, func, None) if func else None
            if isinstance(cls, type) and issubclass(cls, ss.TimePar) and all(isinstance(a, ast.Constant) for a in n.args):
                return ast.Constant(float(ss.solvers.convert(cls(*[a.value for a in n.args]), unit)))
            self.generic_visit(n)
            return n
        def visit_Attribute(self, n):
            if n.attr == 'n':
                return self.visit(n.value)
            elif n.attr == 'frac':
                return ast.BinOp(self.visit(n.value), ast.Div(), ast.Name(names.get('N', 'N')))
            self.generic_visit(n)
            return n
        def visit_Name(self, n):
            return ast.Name(names.get(n.id, n.id))
    return ast.unparse(Sub().visit(node))


def contact_rate(sim, engine, saturated=True):
    """
    The mean contacts per person per unit of time of the sim's mixing (with no network, 1), or None if it has no such number

    A network's edges each transmit at most once in a step, so as the ODE has them, its contacts are fewer in effect
    than it has: (1 − e^(−b·D))/(b·D) of them, for a beta b and a step D (see `ss.solvers.Compartmental.saturation()`);
    these are written, unless `saturated=False`.
    """
    total = 0.0
    for term in sim.mixing.terms:
        if term.kind == 'homogeneous':
            sat = 1.0
            if saturated and term.get('edges') is not None and hasattr(engine, 'saturation'):
                if engine.program is None:
                    engine.init_program() # As the run does, for the exposures that saturation() reads
                sat = engine.saturation(0, term) if engine.prog_exposures else 1.0
            total += term.contacts*float(np.mean(sat))
        elif term.kind == 'matrix' and 'matrix' in term and 'cells' not in term: # The mean over people of the contacts in a row
            rows = np.asarray(term.matrix, dtype=float).sum(axis=1)
            sizes = np.asarray(engine.y).sum(axis=0)
            total += np.average(rows, weights=sizes) if len(rows) == len(sizes) else rows.mean()
        else:
            return None
    return float(total)


def route_beta(sim, betas, unit):
    """ A beta for each route as one beta: their mean weighted by contacts, so that with the total contacts it gives the same transmission """
    key = ss.networks.standardize_netkey
    betas = {key(k):float(ss.solvers.convert(v, unit)) for k,v in betas.items()}
    terms = [(term.contacts, betas.get(key(term.name), 0.0)) for term in sim.mixing.terms if term.kind == 'homogeneous']
    return np.average([b for c,b in terms], weights=[c for c,b in terms])


def year_of(sim):
    """ The expression for the year at time `t` of an export """
    years = sim.t.yearvec
    per_year = (len(years) - 1)*float(sim.t.dt.to(sim.unit_base).value)/(years[-1] - years[0]) # From the timeline, since a year of dates need not be 365 days
    return f'{float(years[0])!r} + t / {float(per_year)!r}'


def seasonal_expr(raw, year):
    """ The factor of an `ss.seasonal`, as an expression in the year: see `ss.seasonal.factor()` """
    amp = float(ss.parameters.const_value(raw.amplitude))
    return f'maximum(0.0, 1 + {amp!r} * cos(2 * pi * ({year} - {raw.peak_frac!r}) / {raw.period_years!r}))'


def timeseries_expr(raw, sim, year, unit):
    """ An `ss.timeseries` as an expression in the year, a piece for each interval: see `ss.timeseries.weights()` """
    years = [float(y) for y in raw.years(sim)]
    vals = [float(ss.solvers.convert(ss.parameters.const_value(v), unit)) for v in raw.values]
    if raw.period is not None: # Fold the time into one period from the first time
        period = raw.period.years if isinstance(raw.period, ss.Duration) else float(raw.period)
        year = f'({year}) - {period!r} * floor((({year}) - {years[0]!r}) / {period!r})'
    out = repr(vals[-1])
    for i in reversed(range(len(vals) - 1)):
        w = f'(({year}) - {years[i]!r}) / {years[i+1] - years[i]!r}'
        w = dict(step=None, linear=w, cosine=f'(1 - cos(pi * {w})) / 2')[raw.interpolation]
        value = repr(vals[i]) if w is None else f'{vals[i]!r} + {vals[i+1] - vals[i]!r} * ({w})'
        out = f'where(({year}) < {years[i+1]!r}, {value}, {out})'
    return f'where(({year}) < {years[0]!r}, {vals[0]!r}, {out})'


def flatten(sim):
    """
    The compartmental form of a sim's declared diseases, as plain data for the exporters

    Returns an `sc.obj` with the unit of time; the compartments and their initial
    numbers; the parameters (numbers in that unit); named expressions (a state with
    several stages is their sum, `N` is the people in a disease, and a parameter that
    varies in time is its formula, with `t` the time); the flows, each with its
    compartments and its rate in total; and what was left out or changed.
    """
    sim, how = compartmental(sim)
    unit = sim.unit_base
    lost = [how] if how else []
    diseases = [mod for mod in sim.diseases.values() if getattr(mod, 'graph', None) is not None]
    many = len(diseases) > 1
    out = sc.obj(name=diseases[0].name if len(diseases) == 1 else 'model', unit=unit, comps=sc.obj(), pars=sc.obj(), exprs=sc.obj(), flows=[], lost=lost, n_states=0, n_transitions=0)

    for mod in diseases:
        g, e = mod.graph, mod.engine
        pre = f'{mod.name}_' if many else ''
        contacts = contact_rate(sim, e) if g.transmissions else 1.0
        word = None
        if contacts is None:
            lost.append(f'changed: the contacts of "{mod.name}" have no single number, so its transmission is written with 1 contact per {unit}')
        elif contacts != 1:
            word = 'contacts' if 'contacts' not in g.pars else 'mixing_contacts'
            out.pars[pre + word] = contacts
        names, flows = ss.display.flow_terms(g, e, contacts=word)
        Y = np.asarray(e.y).sum(axis=1) # Summed over the groups of any stratification
        table = {}
        for i,name in enumerate(names):
            table[name] = pre + name
            out.comps[pre + name] = float(Y[i])
        for state in g.states.values():
            stages = [table[n] for i,n in enumerate(names) if e.comps[i].state is state]
            if len(stages) > 1: # The state is the sum of its stages
                table[state.name] = pre + state.name
                out.exprs[pre + state.name] = ' + '.join(stages)
        table['N'] = pre + 'N'
        if word:
            table[word] = pre + word
        out.exprs[pre + 'N'] = ' + '.join(pre + s.name for s in g.people_states)
        out.n_states += len(g.states)
        out.n_transitions += len(g.transitions)

        # Parameters, as numbers in the sim's unit
        dwells = {id(dw.par): dw for dw in e.dwells.values()}
        for name,p in g.pars.items():
            table[name] = pre + name
            raw = p.raw
            if isinstance(raw, ss.parameters.seasonal) or isinstance(raw, ss.timeseries):
                year = year_of(sim)
                try:
                    if isinstance(raw, ss.timeseries):
                        out.exprs[pre + name] = timeseries_expr(raw, sim, year, unit)
                    else:
                        out.pars[f'{pre}{name}_mean'] = float(ss.solvers.convert(ss.parameters.const_value(raw.mean), unit))
                        out.exprs[pre + name] = f'{pre}{name}_mean * {seasonal_expr(raw, year)}'
                    continue
                except (TypeError, ValueError):
                    pass
            if isinstance(raw, dict):
                try:
                    out.pars[pre + name] = route_beta(sim, raw, unit)
                    continue
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            if isinstance(raw, ss.Dist):
                dw = dwells.get(id(p))
                value = float(dw.mean) if dw is not None else np.nan
                if dw is not None and dw.staged:
                    lost.append(f'changed: the {dw.family} dwell time "{name}" written as {dw.k} exponential stages (Erlang) with its mean, {value:n} {unit}s')
                elif dw is not None and dw.family not in ['expon', None]:
                    lost.append(f'changed: the {dw.family} dwell time "{name}" written as exponential with its mean, {value:n} {unit}s')
            else:
                try:
                    value = float(ss.solvers.convert(raw, unit))
                except (TypeError, ValueError):
                    value = np.nan
                    lost.append(f'not exported: the value of parameter "{name}" ({type(raw).__name__}), which is not a number; written as NaN')
            out.pars[pre + name] = value

        # Flows, each named after its transition (with a number when it is several)
        counts = sc.ddict(int)
        for flow in flows:
            counts[flow.tr.name] += max(len(flow.dst), 1)
        seen = sc.ddict(int)
        for flow in flows:
            src = None if flow.src is None else table[names[flow.src]]
            for dst,term in flow.dst or [(None, flow.term)]:
                name = pre + flow.tr.name
                if counts[flow.tr.name] > 1:
                    seen[flow.tr.name] += 1
                    name = f'{name}_{seen[flow.tr.name]}'
                out.flows.append(sc.obj(name=name, src=src, dst=None if dst is None else table[names[dst]], rate=rename(term, table, unit), transition=flow.tr.name))

    # Anything a rate uses that is not exported
    known = set(out.comps) | set(out.pars) | set(out.exprs)
    for flow in out.flows:
        used = {n.id for n in ast.walk(ast.parse(flow.rate, mode='eval')) if isinstance(n, ast.Name)} - known - set(math_funcs) - set(extrema) - {'pi', 't', 'where'}
        if used:
            lost.append(f'not exported: {sc.strjoin(sorted(used))}, used in the rate of "{flow.name}" ({flow.rate})')

    # What the sim has that these formats do not
    if not diseases:
        lost.append('not exported: the sim has no declared diseases')
    for mod in [mod for key in ['custom', 'demographics', 'diseases'] for mod in sim[key]() if not isinstance(mod, ss.demographics.Flows)]: # Births and deaths declared for the population are, by now, transitions of a disease
        if not any(mod is dis for dis in diseases):
            lost.append(f'not exported: module "{mod.name}" ({type(mod).__name__}), which has no declared transitions')
            continue
        for text in strata(sim, mod):
            lost.append(f'changed: "{mod.name}" is stratified ({text}), written as one group, the sum of its groups, with the base values of its parameters')
        results = [(name, 'result') for name in mod.graph.exprs if name not in mod.graph.marked_results] + [(name, 'outcome') for name in mod.graph.outcomes] # Not those worked out from the infected states, which the model did not write
        for name,kind in sorted(results):
            lost.append(f'not exported: {kind} "{name}"')
        for name in sorted(mod.graph.funcs):
            lost.append(f'not exported: function "{name}" (Python code)')
    for mods,what in [(list(sim.networks()) + getattr(sim, 'mixed', []), 'network'), (sim.interventions(), 'intervention'), (sim.connectors(), 'modifier')]:
        for mod in mods:
            extra = ''
            if what == 'network':
                raw, eff = (contact_rate(sim, diseases[0].engine, saturated=k) if diseases else 0.0 for k in (False, True))
                extra = f' (written as homogeneous mixing with {raw:n} contacts per {unit}' + (f', each transmitting at most once in a step, so {eff:.4g} in effect)' if raw and eff != raw else ')')
            lost.append(f'not exported: {what} "{mod.name}"{extra}')
    if getattr(sim, 'crossings', None):
        lost.append('not exported: the coupling between diseases (each is written on its own)')
    if sim.solver not in ['ode', 'auto'] and how is None:
        lost.append(f'changed: solver="{sim.solver}" is not carried; the model is written as an ODE')
    return out


def names_in(text):
    """ The names an expression uses """
    return {n.id for n in ast.walk(ast.parse(text, mode='eval')) if isinstance(n, ast.Name)}


def in_unit(sim, unit='day'):
    """
    The compartmental form of a sim (see `flatten()`) with its rates in a unit of time, for a format that has one

    The parameters keep the sim's unit, since a number given without one is read in it; each
    rate, and the time, `t`, in the expressions, are converted. The compartmental form also has
    how long the sim runs in that unit (`dur`), its first year (`year`), and its start date
    (`start`, None if it has no dates).
    """
    sim = initialized(sim)
    t = sim.t
    flat = flatten(sim)
    flat.dur = (t.npts - 1)*float(ss.solvers.convert(t.dt, unit))
    flat.year = float(t.yearvec[0])
    flat.start = t.datevec[0] if isinstance(t.datevec[0], ss.date) else None
    if flat.unit != unit:
        up = float(ss.solvers.convert(ss.Duration(1, base=flat.unit), unit)) # The new unit in each of the sim's, e.g. 365 days in a year
        down = float(ss.solvers.convert(ss.Duration(1, base=unit), flat.unit)) # The sim's unit in each of the new one, e.g. 365 days in a year for a sim in days
        per = (lambda text: f'{text} / {up!r}') if up >= 1 else (lambda text: f'{text} * {down!r}') # A rate in the new unit from one in the sim's, and the time in the sim's from the time in the new one
        class Time(ast.NodeTransformer):
            def visit_Name(self, n):
                return ast.parse(f'({per("t")})', mode='eval').body if n.id == 't' else n
        convert = lambda text: ast.unparse(Time().visit(ast.parse(text, mode='eval').body))
        for name,text in flat.exprs.items():
            flat.exprs[name] = convert(text)
        for flow in flat.flows:
            flow.rate = per(f'({convert(flow.rate)})')
        flat.unit = unit
    return flat


def summed(node, flat):
    """ The compartments an expression adds up (a compartment, a named sum of them such as a state's stages or N, or a sum written out), or None if it is not such a sum """
    if isinstance(node, ast.Name) and node.id in flat.exprs:
        node = ast.parse(flat.exprs[node.id], mode='eval').body
    out = []
    for term in terms(node):
        if isinstance(term, ast.Name) and term.id in flat.comps:
            out.append(term.id)
        elif isinstance(term, ast.Name) and term.id in flat.exprs and summed(term, flat) is not None:
            out += summed(term, flat)
        else:
            return None
    return out


def flow_form(flow, flat):
    """
    A flow of `ss.converters.export.flatten()` as one of the usual forms of compartmental models

    Returns an `sc.obj` with the kind, the coefficient (an AST of the factors of the rate that
    do not depend on the compartments, or None), and the compartments of the agents (the infectious):

    - 'linear': coef * source (a transition, or a death if it has no destination)
    - 'infection': coef * source * agents / N, with N every compartment (frequency-dependent)
    - 'density': coef * source * agents (mass action)
    - 'births': coef * N, with N every compartment, and no source
    - 'constant': coef (an import if it has no source, else an absolute flow)
    - 'other': anything else

    Args:
        flow (`sc.obj`): a flow of the flattened sim, with its source, destination, and rate in total
        flat (`sc.obj`): the flattened sim
    """
    every = sorted(flat.comps)
    stateful = set(flat.comps) | {name for name in flat.exprs if summed(ast.Name(name), flat) is not None}
    num, den = factors(ast.parse(flow.rate, mode='eval').body)
    is_source = lambda f: flow.src is not None and isinstance(f, ast.Name) and f.id == flow.src
    while any(map(is_source, num)) and any(map(is_source, den)): # The source above and below the line, as in a total rate written per person (rate / S * S)
        num.pop(next(i for i,f in enumerate(num) if is_source(f)))
        den.pop(next(i for i,f in enumerate(den) if is_source(f)))
    per_source = any(map(is_source, num))
    if per_source:
        num.pop(next(i for i,f in enumerate(num) if is_source(f)))
    above = [summed(f, flat) for f in num]
    below = [summed(f, flat) for f in den]
    coef = product([f for f,s in zip(num, above) if s is None], [f for f,s in zip(den, below) if s is None])
    above, below = [s for s in above if s is not None], [s for s in below if s is not None]
    out = sc.obj(kind='other', coef=None, agents=[])
    if names(coef) & stateful:
        return out
    if per_source and not above and not below:
        out.kind = 'linear'
    elif per_source and len(above) == 1 and len(below) == 1 and sorted(below[0]) == every:
        out.kind, out.agents = 'infection', above[0]
    elif per_source and len(above) == 1 and not below:
        out.kind, out.agents = 'density', above[0]
    elif flow.src is None and len(above) == 1 and sorted(above[0]) == every and not below:
        out.kind = 'births'
    elif not per_source and not above and not below:
        out.kind = 'constant'
    else:
        return out
    out.coef = coef
    return out


def timing(sim):
    """ The times of a sim for an export, as numbers in its unit (start, stop, and dt; a sim with dates starts at 0), and its start date, if any """
    sim = initialized(sim)
    t, unit = sim.t, sim.unit_base
    dt = float(t.dt.to(unit).value)
    start = 0.0 if t.is_absolute else float(t.start.value)
    return sc.obj(unit=unit, start=start, stop=start + (t.npts - 1)*dt, dt=dt, date=t.start if t.is_absolute else None)


def declared_pars(sim):
    """ The parameters (`ss.par`) of a sim's declared diseases, by the name `flatten()` gives them """
    sim = initialized(sim)
    diseases = [mod for mod in sim.diseases.values() if getattr(mod, 'graph', None) is not None]
    many = len(diseases) > 1
    return {(f'{mod.name}_' if many else '') + name:p for mod in diseases for name,p in mod.graph.pars.items()}


def counted(name, exprs):
    """ The compartments a named expression counts (`N`, or a state with several stages), with the expressions in it written out """
    out = set()
    for n in ast.walk(ast.parse(exprs.get(name, name), mode='eval')):
        if isinstance(n, ast.Name):
            out |= counted(n.id, exprs) if n.id in exprs and n.id != name else {n.id}
    return out


#%% SBML

def mathml(node):
    """ A Python expression (an AST node) as a MathML element """
    m = 'http://www.w3.org/1998/Math/MathML'
    def el(tag, text=None):
        e = ET.Element(f'{{{m}}}{tag}')
        if text is not None:
            e.text = f' {text} '
        return e
    def apply(op, args):
        a = el('apply')
        a.append(el(op))
        for arg in args:
            a.append(arg)
        return a
    def visit(n):
        if isinstance(n, ast.Name) and n.id == 't':
            e = el('csymbol', 't')
            e.set('encoding', 'text')
            e.set('definitionURL', 'http://www.sbml.org/sbml/symbols/time')
            return e
        elif isinstance(n, ast.Name) and n.id == 'pi':
            return el('pi')
        elif isinstance(n, ast.Name):
            return el('ci', n.id)
        elif isinstance(n, ast.Constant):
            return el('cn', repr(float(n.value)))
        elif isinstance(n, ast.BinOp):
            ops = {ast.Add:'plus', ast.Sub:'minus', ast.Mult:'times', ast.Div:'divide', ast.Pow:'power'}
            return apply(ops[type(n.op)], [visit(n.left), visit(n.right)])
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return apply('minus', [visit(n.operand)])
        elif isinstance(n, ast.Compare) and isinstance(n.ops[0], ast.Lt):
            return apply('lt', [visit(n.left), visit(n.comparators[0])])
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'where':
            cond, yes, no = n.args
            out = el('piecewise')
            piece = el('piece')
            piece.extend([visit(yes), visit(cond)])
            otherwise = el('otherwise')
            otherwise.append(visit(no))
            out.extend([piece, otherwise])
            return out
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in extrema:
            a, b = n.args
            out = el('piecewise')
            piece = el('piece')
            piece.extend([visit(a), apply(extrema[n.func.id], [visit(a), visit(b)])])
            otherwise = el('otherwise')
            otherwise.append(visit(b))
            out.extend([piece, otherwise])
            return out
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in math_funcs:
            return apply(math_funcs[n.func.id], [visit(a) for a in n.args])
        raise ValueError(f'the expression "{ast.unparse(n)}" has no MathML form here')
    out = el('math')
    out.append(visit(n) if isinstance(node, ast.AST) and not isinstance(node, ast.Expression) else visit(node.body))
    return out


def to_sbml(sim):
    """
    The compartmental form of a sim as an SBML model (Level 3 Version 1): compartments as species, flows as reactions

    Args:
        sim (`ss.Sim`): the sim
    """
    flat = flatten(sim)
    sb = 'http://www.sbml.org/sbml/level3/version1/core'
    ET.register_namespace('', sb)
    ET.register_namespace('math', 'http://www.w3.org/1998/Math/MathML')
    q = lambda tag: f'{{{sb}}}{tag}'
    sub = lambda parent, tag, **attrs: ET.SubElement(parent, q(tag), {k:str(v) for k,v in attrs.items()})
    root = ET.Element(q('sbml'), dict(level='3', version='1'))
    model = sub(root, 'model', id=flat.name if flat.name.isidentifier() else 'model', name=flat.name, timeUnits=flat.unit, substanceUnits='item', extentUnits='item')
    seconds = dict(day=86400, week=7*86400, month=365.25/12*86400, year=365.25*86400)
    units = sub(model, 'listOfUnitDefinitions')
    ud = sub(units, 'unitDefinition', id=flat.unit)
    sub(sub(ud, 'listOfUnits'), 'unit', kind='second', exponent=1, scale=0, multiplier=seconds.get(flat.unit, 1))
    sub(sub(model, 'listOfCompartments'), 'compartment', id='population', size=1, constant='true', spatialDimensions=3, units='dimensionless')
    species = sub(model, 'listOfSpecies')
    for name,init in flat.comps.items():
        sub(species, 'species', id=name, compartment='population', initialAmount=repr(init), hasOnlySubstanceUnits='true', boundaryCondition='false', constant='false')
    params = sub(model, 'listOfParameters')
    for name,value in flat.pars.items():
        sub(params, 'parameter', id=name, value=repr(value), constant='true')
    for name in flat.exprs:
        sub(params, 'parameter', id=name, constant='false')
    lost = list(flat.lost)
    rules = sub(model, 'listOfRules')
    for name,text in flat.exprs.items():
        sub(rules, 'assignmentRule', variable=name).append(mathml(ast.parse(text, mode='eval')))
    reactions = sub(model, 'listOfReactions')
    for flow in flat.flows:
        try:
            law = mathml(ast.parse(flow.rate, mode='eval'))
        except ValueError as E:
            lost.append(f'not exported: the flow "{flow.name}": {E}')
            continue
        r = sub(reactions, 'reaction', id=flow.name, reversible='false', fast='false')
        if flow.src is not None:
            sub(sub(r, 'listOfReactants'), 'speciesReference', species=flow.src, stoichiometry=1, constant='true')
        if flow.dst is not None:
            sub(sub(r, 'listOfProducts'), 'speciesReference', species=flow.dst, stoichiometry=1, constant='true')
        used = {n.id for n in ast.walk(ast.parse(flow.rate, mode='eval')) if isinstance(n, ast.Name)}
        modifiers = [name for name in flat.comps if name in used and name not in [flow.src, flow.dst]] # Other species the law uses, as SBML requires
        if modifiers:
            mods = sub(r, 'listOfModifiers')
            for name in modifiers:
                sub(mods, 'modifierSpeciesReference', species=name)
        sub(r, 'kineticLaw').append(law)
    ET.indent(root, space='  ')
    text = ET.tostring(root, encoding='unicode')
    text = text.replace(' xmlns:math="http://www.w3.org/1998/Math/MathML"', '').replace('<math:math>', '<math xmlns="http://www.w3.org/1998/Math/MathML">') # MathML in its own default namespace, as SBML tools write it
    text = re.sub(r'<(/?)math:', r'<\1', text)
    text = '<?xml version="1.0" encoding="UTF-8"?>\n' + text + '\n'
    summary = f'exported to SBML: {flat.n_states} states, {flat.n_transitions} transitions, as {len(flat.comps)} species and {len(flat.flows)} reactions (rates per {flat.unit})'
    return text, summary, lost


#%% odin

def odin_expr(text):
    """ An expression in R, as odin reads it """
    class Sub(ast.NodeTransformer):
        def visit_Call(self, n):
            self.generic_visit(n)
            if n.func.id == 'where': # R's if, written whole as a name, since Python's expressions have no such form
                cond, yes, no = [ast.unparse(a) for a in n.args]
                return ast.Name(f'(if ({cond}) {yes} else {no})')
            n.func.id = odin_funcs.get(n.func.id, n.func.id)
            return n
        def visit_Name(self, n):
            return ast.Constant(np.pi) if n.id == 'pi' else n # odin has no pi
    return ast.unparse(Sub().visit(ast.parse(text, mode='eval').body)).replace('**', '^')


def to_odin(sim):
    """
    The compartmental form of a sim as an odin model: a `deriv()` equation for each compartment, with each flow a named variable

    Args:
        sim (`ss.Sim`): the sim
    """
    flat = flatten(sim)
    r = odin_expr
    lines = [f'## {flat.name}, exported from Starsim {ss.__version__} (tested with {ss.converters.core.tested_with("odin")}); time is in {flat.unit}s', '']
    taken = set(flat.comps) | set(flat.pars) | set(flat.exprs)
    flow_names = []
    for flow in flat.flows: # Each flow is its own variable, so the equations pair up when read back
        name = flow.name if flow.name not in taken else f'{flow.name}_flow'
        taken.add(name)
        flow_names.append(name)
    for comp in flat.comps:
        terms = [f'+ {n}' for f,n in zip(flat.flows, flow_names) if f.dst == comp] + [f'- {n}' for f,n in zip(flat.flows, flow_names) if f.src == comp]
        rhs = ' '.join(terms).lstrip('+ ') if terms else '0'
        if rhs.startswith('- '):
            rhs = '-' + rhs[2:]
        lines.append(f'deriv({comp}) <- {rhs}')
    lines.append('')
    for comp,init in flat.comps.items():
        lines.append(f'initial({comp}) <- {init:.15g}')
    lines.append('')
    for flow,name in zip(flat.flows, flow_names):
        lines.append(f'{name} <- {r(flow.rate)}')
    for name,text in flat.exprs.items():
        lines.append(f'{name} <- {r(text)}')
    lines.append('')
    for name,value in flat.pars.items():
        lines.append(f'{name} <- user({"NA" if np.isnan(value) else f"{value:.15g}"})')
    summary = f'exported to odin: {flat.n_states} states, {flat.n_transitions} transitions, as {len(flat.comps)} compartments and {len(flat.flows)} flows (time in {flat.unit}s)'
    return '\n'.join(lines) + '\n', summary, list(flat.lost)


#%% Exporting

exporters = dict(
    sbml     = to_sbml,
    odin     = to_odin,
    epimodel = to_epimodel,
    siminf   = to_siminf,
    summer2  = to_summer2,
    epydemix = to_epydemix,
    epipack  = to_epipack,
    epymorph = to_epymorph,
    memilio  = to_memilio,
    atomica  = to_atomica,
    camdl    = to_camdl,
    emulsion = to_emulsion,
    flepimop = to_flepimop,
    json     = to_json,
)

extensions = dict( # The extension of each format's files, for `starsim convert` (Starsim's own is its code)
    starsim  = '.py',
    sbml     = '.xml',
    odin     = '.R',
    epimodel = '.R',
    siminf   = '.R',
    summer2  = '.py',
    epydemix = '.py',
    epipack  = '.py',
    epymorph = '.py',
    memilio  = '.py',
    atomica  = '.xlsx',
    camdl    = '.camdl',
    emulsion = '.yaml',
    flepimop = '.yml',
    json     = '.json',
)


def export_to(sim, format, filename=None, verbose=None):
    """
    Write a sim in another format (`sim.save()` and `ss.save()` write Starsim's own, a gzipped pickle that `ss.load()` reads)

    What the format cannot carry is listed as `ss.errors.LossyExport`; the export still
    writes everything it can.

    Args:
        sim (`ss.Sim`): the sim, with declared diseases
        format (str): one of `ss.converters.exporters` (e.g. 'sbml', 'odin', 'epimodel', 'summer2')
        filename (str): the file to write, if any
        verbose (bool): whether to print what was exported (by default, unless something was lost)

    Returns:
        The text of the export (bytes for Atomica's workbook; for flepiMoP, a dict of its files by name, the config first)

    Examples:
        ```python
        ss.export_to(sim, 'sbml', filename='sir.xml')
        print(ss.export_to(sim, 'odin'))
        ```
    """
    key = str(format).lower()
    if key not in exporters:
        errormsg = f'Cannot export to format "{format}"; the formats are {sc.strjoin(exporters)}'
        raise ValueError(errormsg)
    ss.converters.check_version(key)
    text, summary, lost = exporters[key](sim)
    if lost:
        ss.warn(f'{summary}\n' + '\n'.join(f'  {line}' for line in lost), ss.errors.LossyExport)
    elif verbose is not False:
        print(summary)
    if filename is not None:
        files = text if isinstance(text, dict) else {None: text} # Several files (flepiMoP's config and its data): the first is the filename, the rest beside it
        for i,(name,content) in enumerate(files.items()):
            path = sc.path(filename) if i == 0 else sc.path(filename).parent/name
            if isinstance(content, bytes): # A workbook (Atomica's)
                path.write_bytes(content)
            else:
                sc.savetext(path, content)
    return text


#%% Exporting runs

hub_quantiles = [0.01, 0.025] + [round(q, 2) for q in np.arange(0.05, 0.951, 0.05)] + [0.975, 0.99] # The 23 quantiles of the US forecast hubs

def to_hub(msim, targets, origin_date=None, by=None, every='week', output_type='quantile', quantiles=hub_quantiles, location=None):
    """
    The runs of a MultiSim as a submission to a hub (in the hubverse format): each target, by scenario, location, and period

    Each scenario (the label of its sim) is its own `scenario_id`, and the periods are counted from
    `origin_date`. A target's name says what it is: one starting "cum " is the total since the origin
    date, one starting "peak size " is the largest value of a period (with no horizon), and any other is
    a flow's total over each period, or a count at its end.

    Args:
        msim (`ss.MultiSim`): the runs
        targets (dict): the hub's name (or a list of names) for each result, e.g. `dict(incidHosp=['inc hosp', 'cum hosp', 'peak size hosp'])`
        origin_date (date): the date the periods are counted from (default: the start of the runs); for runs without dates, a time, which the `origin_date` column then holds
        by (str): the dimension whose levels are the locations (default: one location, the total)
        every (str): the period: 'day' or 'week' (of the sim's steps, which must then be days)
        output_type (str/list): 'quantile' (of the runs of each scenario), 'sample' (each run, by its position in its scenario), or both
        quantiles (list): the quantiles (default: the 23 that the US hubs ask for)
        location (str): the name of the single location, if `by` is not given

    Returns a long dataframe with columns `origin_date`, `scenario_id`, `location`, `target`, `horizon`,
    `output_type`, `output_type_id`, and `value`.

    Example:
        ```python
        msim = ss.parallel(baseline=s1, lockdown=s2, n_runs=20)
        hub = ss.converters.to_hub(msim, targets=dict(infection=['inc inf', 'cum inf']), origin_date='2020-03-01', by='patch')
        ```
    """
    msim._check_ran('to_hub')
    steps = dict(day=1, week=7)[every]
    output_type = sc.tolist(output_type)
    if not set(output_type) <= {'quantile', 'sample'}:
        errormsg = f'The output type must be "quantile" or "sample", not {output_type}'
        raise ValueError(errormsg)
    found = {key:res for modname,key,res in msim.results.walk() if modname is not None}
    rows = []
    for key,names in targets.items():
        if key not in found:
            errormsg = f'No result called "{key}"; the results are {sc.strjoin(found)}'
            raise KeyError(errormsg)
        res = found[key]
        dates = list(res.timevec)
        dated = isinstance(dates[0], ss.date)
        origin = dates[0] if origin_date is None else ss.date(origin_date) if dated else float(origin_date) # Without dates, a time in the sim's unit
        i0 = next((i for i,d in enumerate(dates) if d >= origin), len(dates)) # The values after the origin
        places = res.by(by)._columns() if by is not None else {location or 'total':res}
        for place,sub in places.items():
            vals = sub.values[:, i0+1:]
            n = vals.shape[1]//steps
            if sub.kind == 'flow':
                periods = vals[:, :n*steps].reshape(len(vals), n, steps).sum(axis=2)
            else:
                periods = vals[:, steps-1:n*steps:steps]
            for target in sc.tolist(names):
                if target.startswith('cum '):
                    if sub.kind != 'flow':
                        errormsg = f'The target "{target}" is a cumulative total, but "{key}" is a count, not a flow'
                        raise ValueError(errormsg)
                    values, horizon = periods.cumsum(axis=1), np.arange(1, n+1)
                elif target.startswith('peak size '):
                    values, horizon = periods.max(axis=1, keepdims=True), [pd.NA]
                else:
                    values, horizon = periods, np.arange(1, n+1)
                for scenario,inds in sub.scenarios().items():
                    common = dict(origin_date=origin.strftime('%Y-%m-%d') if dated else float(origin), scenario_id=scenario, location=str(place), target=target, horizon=pd.array(horizon, dtype='Int64'))
                    if 'quantile' in output_type:
                        for q,qvals in zip(quantiles, np.quantile(values[inds], quantiles, axis=0)):
                            rows.append(pd.DataFrame(dict(**common, output_type='quantile', output_type_id=q, value=qvals)))
                    if 'sample' in output_type:
                        for j,ind in enumerate(inds):
                            rows.append(pd.DataFrame(dict(**common, output_type='sample', output_type_id=j, value=values[ind])))
    return pd.concat(rows, ignore_index=True)
